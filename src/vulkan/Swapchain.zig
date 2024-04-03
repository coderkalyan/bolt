const std = @import("std");
const builtin = @import("builtin");
const Instance = @import("Instance.zig");
const App = @import("../App.zig");
const Allocator = std.mem.Allocator;

const c = Instance.c;

const Swapchain = @This();

gpa: Allocator,
vulkan: *const Instance,
app: *App,

swapchain: c.VkSwapchainKHR,
images: []const c.VkImage,
format: c.VkFormat,
extent: c.VkExtent2D,
views: []const c.VkImageView,

ds_swapchain: []const c.VkDescriptorSet,
image_available: c.VkSemaphore,
render_finished: c.VkSemaphore,
in_flight: c.VkFence,
ds_cells: c.VkDescriptorSet,

pub fn init(gpa: Allocator, vulkan: *const Instance) !Swapchain {
    var arena_allocator = std.heap.ArenaAllocator.init(gpa);
    defer arena_allocator.deinit();
    const arena = arena_allocator.allocator();

    // query swapchain support parameters
    var caps: c.VkSurfaceCapabilitiesKHR = undefined;
    if (c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vulkan.phydev, vulkan.surface, &caps) != c.VK_SUCCESS) {
        return error.VkGetPhysicalDeviceSurfaceCapabilitiesFailed;
    }

    var format_count: u32 = 0;
    if (c.vkGetPhysicalDeviceSurfaceFormatsKHR(vulkan.phydev, vulkan.surface, &format_count, null) != c.VK_SUCCESS) {
        return error.VkGetPhysicalDeviceSurfaceFormatsFailed;
    }
    if (format_count == 0) return error.VkSwapchainNoAvailableFormats;

    const formats = try arena.alloc(c.VkSurfaceFormatKHR, format_count);
    if (c.vkGetPhysicalDeviceSurfaceFormatsKHR(vulkan.phydev, vulkan.surface, &format_count, formats.ptr) != c.VK_SUCCESS) {
        return error.VkGetPhysicalDeviceSurfaceFormatsFailed;
    }

    var mode_count: u32 = 0;
    if (c.vkGetPhysicalDeviceSurfacePresentModesKHR(vulkan.phydev, vulkan.surface, &mode_count, null) != c.VK_SUCCESS) {
        return error.VkGetPhysicalDeviceSurfacePresentModesFailed;
    }
    if (mode_count == 0) return error.VkSwapchainNoAvailableModes;

    const modes = try arena.alloc(c.VkPresentModeKHR, mode_count);
    if (c.vkGetPhysicalDeviceSurfacePresentModesKHR(vulkan.phydev, vulkan.surface, &mode_count, modes.ptr) != c.VK_SUCCESS) {
        return error.VkGetPhysicalDeviceSurfacePresentModesFailed;
    }

    // choose swapchain format and initialize swapchain
    const format = format: {
        for (formats) |format| {
            if (format.format == c.VK_FORMAT_R8G8B8A8_UINT) {
                break :format format;
            }
        }

        // TODO: the fallback case may not support writing to
        // swapchain in compute and presentation
        break :format formats[0];
    };
    std.debug.print("format: {}\n", .{format.format});

    // guaranteed to exist and is energy efficient
    const mode = c.VK_PRESENT_MODE_FIFO_KHR;
    // const mode = c.VK_PRESENT_MODE_MAILBOX_KHR;

    const extent: c.VkExtent2D = if (caps.currentExtent.width != std.math.maxInt(u32)) extent: {
        break :extent caps.currentExtent;
    } else extent: {
        const size = vulkan.app.terminal.size;
        break :extent .{
            .width = size.width,
            .height = size.height,
        };
    };

    const max_image_count = caps.maxImageCount;
    // choose min count + 1, bounded by max count (if present)
    var image_count = @min(
        if (max_image_count == 0) std.math.maxInt(u32) else max_image_count,
        caps.minImageCount + 1,
    );

    const distinct = vulkan.comp_index != vulkan.pres_index;
    const indices: []const u32 = &.{ vulkan.comp_index, vulkan.pres_index };

    const swapchain_info: c.VkSwapchainCreateInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = null,
        .flags = 0,
        .surface = vulkan.surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | c.VK_IMAGE_USAGE_STORAGE_BIT,
        // TODO: why is this breaking
        // .imageSharingMode = if (distinct) c.VK_SHARING_MODE_CONCURRENT else c.VK_SHARING_MODE_EXCLUSIVE,
        .imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = if (distinct) @intCast(indices.len) else 0,
        .pQueueFamilyIndices = if (distinct) indices.ptr else null,
        .preTransform = caps.currentTransform,
        .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = mode,
        .clipped = c.VK_TRUE,
        .oldSwapchain = @ptrCast(c.VK_NULL_HANDLE),
    };

    var swapchain: c.VkSwapchainKHR = undefined;
    if (c.vkCreateSwapchainKHR(vulkan.device, &swapchain_info, null, &swapchain) != c.VK_SUCCESS) {
        return error.VkCreateSwapchainFailed;
    }
    errdefer c.vkDestroySwapchainKHR(vulkan.device, swapchain, null);

    // obtain swapchain image handles
    if (c.vkGetSwapchainImagesKHR(vulkan.device, swapchain, &image_count, null) != c.VK_SUCCESS) {
        return error.VkGetSwapchainImagesFailed;
    }

    const images = try gpa.alloc(c.VkImage, image_count);
    errdefer gpa.free(images);
    if (c.vkGetSwapchainImagesKHR(vulkan.device, swapchain, &image_count, images.ptr) != c.VK_SUCCESS) {
        return error.VkGetSwapchainImagesFailed;
    }

    // bind image views for swapchain images
    const views = try gpa.alloc(c.VkImageView, images.len);
    errdefer gpa.free(views);
    for (images, 0..) |image, i| {
        const view_info: c.VkImageViewCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .image = image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = format.format,
            .components = .{
                .r = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = c.VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        if (c.vkCreateImageView(vulkan.device, &view_info, null, &views[i]) != c.VK_SUCCESS) {
            return error.VkCreateImageViewFailed;
        }
        errdefer c.vkDestroyImageView(vulkan.device, views[i], null);
    }

    // create synchronization primitives
    const semaphore_info: c.VkSemaphoreCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
    };

    const fence_info: c.VkFenceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = null,
        .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
    };

    var image_available: c.VkSemaphore = undefined;
    if (c.vkCreateSemaphore(vulkan.device, &semaphore_info, null, &image_available) != c.VK_SUCCESS) {
        return error.VkCreateSemaphoreFailed;
    }
    errdefer c.vkDestroySemaphore(vulkan.device, image_available, null);

    var render_finished: c.VkSemaphore = undefined;
    if (c.vkCreateSemaphore(vulkan.device, &semaphore_info, null, &render_finished) != c.VK_SUCCESS) {
        return error.VkCreateSemaphoreFailed;
    }
    errdefer c.vkDestroySemaphore(vulkan.device, render_finished, null);

    var in_flight: c.VkFence = undefined;
    if (c.vkCreateFence(vulkan.device, &fence_info, null, &in_flight) != c.VK_SUCCESS) {
        return error.VkCreateFenceFailed;
    }
    errdefer c.vkDestroyFence(vulkan.device, in_flight, null);

    // bind swapchain to each of the appropriate descriptor sets
    // this only gets changed when the swapchain is re-initialized on resize
    // (this function is called again)
    const ds_swapchain = try gpa.alloc(c.VkDescriptorSet, image_count);
    errdefer gpa.free(ds_swapchain);
    var i: u32 = 0;
    while (i < image_count) : (i += 1) {
        const ds_info: c.VkDescriptorSetAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = vulkan.ds_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &vulkan.ds_layout_swapchain,
        };

        if (c.vkAllocateDescriptorSets(vulkan.device, &ds_info, &ds_swapchain[i]) != c.VK_SUCCESS) {
            return error.VkAllocateDescriptorSetsFailed;
        }

        const ds_swapchain_write: c.VkWriteDescriptorSet = .{
            .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = null,
            .dstSet = ds_swapchain[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .pBufferInfo = null,
            .pImageInfo = &c.VkDescriptorImageInfo{
                .sampler = @ptrCast(c.VK_NULL_HANDLE),
                .imageView = views[i],
                .imageLayout = c.VK_IMAGE_LAYOUT_GENERAL,
            },
            .pTexelBufferView = null,
        };

        c.vkUpdateDescriptorSets(vulkan.device, 1, &ds_swapchain_write, 0, null);
    }
    // TODO: this doesn't free in middle of loop
    errdefer _ = c.vkFreeDescriptorSets(vulkan.device, vulkan.ds_pool, image_count, ds_swapchain.ptr);

    const ds_cells_info: c.VkDescriptorSetAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = null,
        .descriptorPool = vulkan.ds_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vulkan.ds_layout_cells,
    };

    var ds_cells: c.VkDescriptorSet = undefined;
    if (c.vkAllocateDescriptorSets(vulkan.device, &ds_cells_info, &ds_cells) != c.VK_SUCCESS) {
        return error.VkAllocateDescriptorSetsFailed;
    }
    errdefer _ = c.vkFreeDescriptorSets(vulkan.device, vulkan.ds_pool, 1, &ds_cells);

    // upload test buffer as SSBO for compute shader
    const lipsum = @embedFile("../lipsum.txt");
    const size = @sizeOf(u8) * lipsum.len;
    const aligned_size = (size + 256 - 1) & ~(@as(usize, 256) - 1); // TODO: necessary?

    var cells_ssbo_staging: c.VkBuffer = undefined;
    var cells_mem_staging: c.VkDeviceMemory = undefined;
    try vulkan.createBuffer(
        aligned_size,
        c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &cells_ssbo_staging,
        &cells_mem_staging,
    );
    defer c.vkDestroyBuffer(vulkan.device, cells_ssbo_staging, null);
    defer c.vkFreeMemory(vulkan.device, cells_mem_staging, null);

    var data: [*]u8 = undefined;
    _ = c.vkMapMemory(vulkan.device, cells_mem_staging, 0, lipsum.len, 0, @ptrCast(&data));
    @memcpy(data, lipsum);
    c.vkUnmapMemory(vulkan.device, cells_mem_staging);

    var cells_ssbo: c.VkBuffer = undefined;
    var cells_mem: c.VkDeviceMemory = undefined;
    try vulkan.createBuffer(
        aligned_size,
        c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &cells_ssbo,
        &cells_mem,
    );
    vulkan.copyBuffer(cells_ssbo_staging, cells_ssbo, aligned_size);

    const ds_cells_write: c.VkWriteDescriptorSet = .{
        .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = null,
        .dstSet = ds_cells,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &c.VkDescriptorBufferInfo{
            .buffer = cells_ssbo,
            .offset = 0,
            .range = aligned_size,
        },
        .pImageInfo = null,
        .pTexelBufferView = null,
    };

    c.vkUpdateDescriptorSets(vulkan.device, 1, &ds_cells_write, 0, null);

    return .{
        .gpa = gpa,
        .vulkan = vulkan,
        .app = vulkan.app,
        .swapchain = swapchain,
        .images = images,
        .format = format.format,
        .extent = extent,
        .views = views,
        .ds_swapchain = ds_swapchain,
        .ds_cells = ds_cells,
        .image_available = image_available,
        .render_finished = render_finished,
        .in_flight = in_flight,
    };
}

pub fn deinit(self: *Swapchain) void {
    const vulkan = self.vulkan;
    if (vulkan.device == null) return;

    // wait for all frames in flight to finish rendering
    _ = c.vkDeviceWaitIdle(vulkan.device);

    for (self.views) |view| c.vkDestroyImageView(vulkan.device, view, null);
    self.gpa.free(self.views);
    self.gpa.free(self.images);
    c.vkDestroySwapchainKHR(vulkan.device, self.swapchain, null);

    _ = c.vkFreeDescriptorSets(vulkan.device, vulkan.ds_pool, @intCast(self.ds_swapchain.len), self.ds_swapchain.ptr);
    self.gpa.free(self.ds_swapchain);
    _ = c.vkFreeDescriptorSets(vulkan.device, vulkan.ds_pool, 1, &self.ds_cells);

    c.vkDestroySemaphore(vulkan.device, self.image_available, null);
    c.vkDestroySemaphore(vulkan.device, self.render_finished, null);
    c.vkDestroyFence(vulkan.device, self.in_flight, null);
}

fn recordCommandBuffer(
    self: *Swapchain,
    image_index: u32,
) !void {
    // render command buffer:
    // * bind compute pipeline
    // * bind cell SSBO (TODO)
    // * push constants for general terminal information
    // * image memory barrier: undefined -> general for compute
    // * dispatch compute
    // * image memory barrier: general -> present_src for presentation
    const vulkan = self.vulkan;
    const begin_info: c.VkCommandBufferBeginInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = null,
        .flags = 0,
        .pInheritanceInfo = null,
    };
    if (c.vkBeginCommandBuffer(vulkan.cmd_buffer, &begin_info) != c.VK_SUCCESS) {
        return error.VkBeginCommandBufferFailed;
    }

    c.vkCmdBindPipeline(vulkan.cmd_buffer, c.VK_PIPELINE_BIND_POINT_COMPUTE, vulkan.pipeline);
    c.vkCmdBindDescriptorSets(
        vulkan.cmd_buffer,
        c.VK_PIPELINE_BIND_POINT_COMPUTE,
        vulkan.pipeline_layout,
        0,
        1,
        &self.ds_swapchain[image_index],
        0,
        0,
    );
    c.vkCmdBindDescriptorSets(
        vulkan.cmd_buffer,
        c.VK_PIPELINE_BIND_POINT_COMPUTE,
        vulkan.pipeline_layout,
        1,
        1,
        &self.ds_cells,
        0,
        0,
    );
    c.vkCmdPushConstants(
        vulkan.cmd_buffer,
        vulkan.pipeline_layout,
        c.VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        @sizeOf(App.Terminal),
        &self.app.terminal,
    );

    c.vkCmdPipelineBarrier(
        vulkan.cmd_buffer,
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        c.VK_DEPENDENCY_BY_REGION_BIT, // TODO: correct?
        0,
        null,
        0,
        null,
        1,
        &c.VkImageMemoryBarrier{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = 0,
            .dstAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = c.VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = vulkan.comp_index,
            .dstQueueFamilyIndex = vulkan.comp_index,
            .image = self.images[image_index],
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        },
    );

    const width = self.app.terminal.size.width;
    const height = self.app.terminal.size.height;
    c.vkCmdDispatch(vulkan.cmd_buffer, (width + 7) / 8, (height + 7) / 8, 1);

    c.vkCmdPipelineBarrier(
        vulkan.cmd_buffer,
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        c.VK_DEPENDENCY_BY_REGION_BIT, // TODO: correct?
        0,
        null,
        0,
        null,
        1,
        &c.VkImageMemoryBarrier{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = 0,
            .oldLayout = c.VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = vulkan.comp_index,
            .dstQueueFamilyIndex = vulkan.comp_index,
            .image = self.images[image_index],
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        },
    );

    if (c.vkEndCommandBuffer(vulkan.cmd_buffer) != c.VK_SUCCESS) {
        return error.VkCommandBufferRecordFailed;
    }
}

pub fn drawFrame(self: *Swapchain) !void {
    const vulkan = self.vulkan;
    _ = c.vkWaitForFences(vulkan.device, 1, &self.in_flight, c.VK_TRUE, std.math.maxInt(u64));
    _ = c.vkResetFences(vulkan.device, 1, &self.in_flight);

    var image_index: u32 = 0;
    const result = c.vkAcquireNextImageKHR(
        vulkan.device,
        self.swapchain,
        std.math.maxInt(u64),
        self.image_available,
        @ptrCast(c.VK_NULL_HANDLE),
        &image_index,
    );
    switch (result) {
        c.VK_ERROR_OUT_OF_DATE_KHR => return error.SwapchainOutOfDate,
        c.VK_SUCCESS => {},
        else => return error.VkAcquireImageFailed,
    }

    const start = std.time.microTimestamp();
    _ = c.vkResetCommandBuffer(vulkan.cmd_buffer, 0);
    const end = std.time.microTimestamp();
    // TODO: cache this?
    try self.recordCommandBuffer(image_index);

    const wait_semaphores: []const c.VkSemaphore = &.{self.image_available};
    const wait_stages: []const c.VkPipelineStageFlags = &.{
        // c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    };
    const signal_semaphores: []const c.VkSemaphore = &.{self.render_finished};
    const submit_info: c.VkSubmitInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = null,
        .waitSemaphoreCount = @intCast(wait_semaphores.len),
        .pWaitSemaphores = wait_semaphores.ptr,
        .pWaitDstStageMask = wait_stages.ptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &vulkan.cmd_buffer,
        .signalSemaphoreCount = @intCast(signal_semaphores.len),
        .pSignalSemaphores = signal_semaphores.ptr,
    };

    if (c.vkQueueSubmit(vulkan.comp_queue, 1, &submit_info, self.in_flight) != c.VK_SUCCESS) {
        return error.VkSubmitDrawFailed;
    }

    _ = c.vkWaitSemaphores(
        vulkan.device,
        &c.VkSemaphoreWaitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext = null,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &self.render_finished,
            .pValues = &@as(u64, 0),
        },
        std.math.maxInt(u64),
    );

    std.debug.print("frame time: {}\n", .{end - start});
    const present_info: c.VkPresentInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = null,
        .waitSemaphoreCount = @intCast(signal_semaphores.len),
        .pWaitSemaphores = signal_semaphores.ptr,
        .swapchainCount = 1,
        .pSwapchains = &self.swapchain,
        .pImageIndices = &image_index,
        .pResults = null,
    };
    _ = c.vkQueuePresentKHR(vulkan.pres_queue, &present_info);
}
