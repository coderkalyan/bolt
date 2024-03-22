const std = @import("std");
const builtin = @import("builtin");
const App = @import("App.zig");
const Wayland = @import("Wayland.zig");
const GlyphCache = @import("GlyphCache.zig");
const Allocator = std.mem.Allocator;

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
    @cInclude("vulkan/vulkan_wayland.h");
    @cInclude("vulkan/vk_enum_string_helper.h");
    @cInclude("pixman.h");
});

const Swapchain = @This();

vulkan: *const VulkanInstance,

swapchain: c.VkSwapchainKHR,
images: []const c.VkImage,
format: c.VkFormat,
extent: c.VkExtent2D,
views: []const c.VkImageView,

ds_set_layout: c.VkDescriptorSetLayout,
ds_set: c.VkDescriptorSet,
image_available: c.VkSemaphore,
render_finished: c.VkSemaphore,
in_flight: c.VkFence,

pub fn init(app: *App) !Vulkan {
    const gpa = app.gpa;

    const descriptor_pool = try createDescriptorPool(device);
    const pipeline_info = try createRenderPipeline(gpa, device);

    // const atlas_layout_binding: c.VkDescriptorSetLayoutBinding = .{
    //     .binding = 0,
    //     .descriptorCount = 1,
    //     .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    //     .pImmutableSamplers = null,
    //     .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
    // };
    // const dest_layout_binding: c.VkDescriptorSetLayoutBinding = .{
    //     .binding = 1,
    //     .descriptorCount = 1,
    //     .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    //     .pImmutableSamplers = null,
    //     .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
    // };
    // const bindings: []const c.VkDescriptorSetLayoutBinding = &.{
    //     atlas_layout_binding,
    //     dest_layout_binding,
    // };
    // const layout_info: c.VkDescriptorSetLayoutCreateInfo = .{
    //     .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    //     .pNext = null,
    //     .flags = 0,
    //     .bindingCount = bindings.len,
    //     .pBindings = bindings.ptr,
    // };
    // var descriptor_set_layout: c.VkDescriptorSetLayout = undefined;
    // if (c.vkCreateDescriptorSetLayout(device, &layout_info, null, &descriptor_set_layout) != c.VK_SUCCESS) {
    //     return error.VkCreateDescriptorSetLayoutFailed;
    // }

    // const descriptor_alloc_info: c.VkDescriptorSetAllocateInfo = .{
    //     .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    //     .pNext = null,
    //     .descriptorPool = descriptor_pool,
    //     .descriptorSetCount = 1,
    //     .pSetLayouts = &descriptor_set_layout,
    // };
    // var descriptor_set: c.VkDescriptorSet = undefined;
    // if (c.vkAllocateDescriptorSets(device, &descriptor_alloc_info, &descriptor_set) != c.VK_SUCCESS) {
    //     return error.VkAllocateDescriptorSetsFailed;
    // }

    var graphics_queue: c.VkQueue = undefined;
    var presentation_queue: c.VkQueue = undefined;
    var compute_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, queue_families.graphics, 0, &graphics_queue);
    c.vkGetDeviceQueue(device, queue_families.presentation, 0, &presentation_queue);
    c.vkGetDeviceQueue(device, queue_families.compute, 0, &compute_queue);

    var vulkan: Vulkan = .{
        .gpa = gpa,
        .app = app,
        .instance = instance,
        .surface = surface,
        .physical_device = physical_device,
        .device = device,
        .queue_families = queue_families,
        .graphics_queue = graphics_queue,
        .presentation_queue = presentation_queue,
        .compute_queue = compute_queue,
        .command_pool = command_pool,
        .sync_objects = sync_objects,
        .swapchain = undefined,
        .pipeline_layout = pipeline_info.layout,
        .pipeline = pipeline_info.pipeline,
        .global_set_layout = pipeline_info.set_layout,
        .global_sets = &.{},
        .command_buffer = null,
        .descriptor_pool = descriptor_pool,
        // .descriptor_set_layout = descriptor_set_layout,
        // .descriptor_set = descriptor_set,
    };

    return vulkan;
}

pub fn init(gpa: Allocator, vulkan: *const VulkanInstance) !Instance {
    var arena_allocator = std.heap.ArenaAllocator.init(gpa);
    defer arena_allocator.deinit();
    const arena = arena_allocator.allocator();

    // choose swapchain format and initialize swapchain
    const format = format: {
        for (support.formats) |format| {
            if (format.format == c.VK_FORMAT_R8G8B8A8_UINT) {
                break :format format;
            }
        }
        // TODO: the fallback case may not support writing to
        // swapchain in compute and presentation
        break :format support.formats[0];
    };

    // guaranteed to exist and is energy efficient
    const mode = c.VK_PRESENT_MODE_FIFO_KHR;

    const extent = chooseSwapExtent(&support.capabilities, self.app);
    const max_image_count = support.capabilities.maxImageCount;
    // choose min count + 1, bounded by max count (if present)
    var image_count = @min(
        if (max_image_count == 0) std.math.maxInt(u32) else max_image_count,
        support.capabilities.minImageCount + 1,
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
        .queueFamilyIndexCount = if (distinct) indices.len else 0,
        .pQueueFamilyIndices = if (distinct) indices.ptr else null,
        .preTransform = support.capabilities.currentTransform,
        .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = mode,
        .clipped = c.VK_TRUE,
        .oldSwapchain = @ptrCast(c.VK_NULL_HANDLE),
    };

    var swapchain: c.VkSwapchainKHR = undefined;
    if (c.vkCreateSwapchainKHR(vulkan.device, &swapchain_info, null, &swapchain) != c.VK_SUCCESS) {
        return error.VkCreateSwapchainFailed;
    }

    // obtain swapchain image handles
    if (c.vkGetSwapchainImagesKHR(vulkan.device, swapchain, &image_count, null) != c.VK_SUCCESS) {
        return error.VkGetSwapchainImagesFailed;
    }

    const images = try gpa.alloc(c.VkImage, image_count);
    if (c.vkGetSwapchainImagesKHR(vulkan.device, swapchain, &image_count, images.ptr) != c.VK_SUCCESS) {
        return error.VkGetSwapchainImagesFailed;
    }

    // bind image views for swapchain images
    const views = try gpa.alloc(c.VkImageView, images.len);
    for (images, 0..) |image, i| {
        const view_info: c.VkImageViewCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .image = image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
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
    if (c.vkCreateSemaphore(device, &semaphore_info, null, &image_available) != c.VK_SUCCESS) {
        return error.VkCreateSemaphoreFailed;
    }

    var render_finished: c.VkSemaphore = undefined;
    if (c.vkCreateSemaphore(device, &semaphore_info, null, &render_finished) != c.VK_SUCCESS) {
        return error.VkCreateSemaphoreFailed;
    }

    var in_flight: c.VkFence = undefined;
    if (c.vkCreateFence(device, &fence_info, null, &in_flight) != c.VK_SUCCESS) {
        return error.VkCreateFenceFailed;
    }

    return .{
        .vulkan = vulkan,
        .swapchain = swapchain,
        .images = images,
        .format = format,
        .extent = extent,
        .views = views,
        .image_available = image_available,
        .render_finished = render_finished,
        .in_flight = in_flight,
    };

    self.global_sets = try self.createGlobalSets();
    try self.bindGlobalSets();

    self.command_buffer = try self.createCommandBuffer();
}

fn querySwapchainSupport(
    gpa: Allocator,
    device: c.VkPhysicalDevice,
    surface: c.VkSurfaceKHR,
) !SwapChainSupport {
    var capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
    _ = c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities);

    var format_count: u32 = 0;
    _ = c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, null);
    const formats = try gpa.alloc(c.VkSurfaceFormatKHR, format_count);
    if (format_count > 0) {
        _ = c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, formats.ptr);
    }

    var mode_count: u32 = 0;
    _ = c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, null);
    const modes = try gpa.alloc(c.VkPresentModeKHR, mode_count);
    if (mode_count > 0) {
        _ = c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, modes.ptr);
    }

    return .{
        .capabilities = capabilities,
        .formats = formats,
        .modes = modes,
    };
}

fn createSwapchain(self: *Vulkan) !Swapchain {
    const gpa = self.gpa;

    const support = try querySwapchainSupport(gpa, self.physical_device, self.surface);
    defer gpa.free(support.formats);
    defer gpa.free(support.modes);

    const format = format: {
        for (support.formats) |format| {
            if (format.format == c.VK_FORMAT_R8G8B8A8_UINT) {
                break :format format;
            }
        }
        break :format support.formats[0];
    };

    // guaranteed to exist and is energy efficient
    const mode = c.VK_PRESENT_MODE_FIFO_KHR;

    const extent = chooseSwapExtent(&support.capabilities, self.app);
    const max_image_count = support.capabilities.maxImageCount;
    const image_count = @min(
        if (max_image_count == 0) std.math.maxInt(u32) else max_image_count,
        support.capabilities.minImageCount + 1,
    );

    const distinct = self.queue_families.graphics != self.queue_families.presentation;
    const queue_family_indices: []const u32 = &.{
        self.queue_families.graphics,
        self.queue_families.presentation,
    };
    const create_info: c.VkSwapchainCreateInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = null,
        .flags = 0,
        .surface = self.surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | c.VK_IMAGE_USAGE_STORAGE_BIT,
        // TODO: why is this breaking
        // .imageSharingMode = if (distinct) c.VK_SHARING_MODE_CONCURRENT else c.VK_SHARING_MODE_EXCLUSIVE,
        .imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = if (distinct) 2 else 0,
        .pQueueFamilyIndices = if (distinct) queue_family_indices.ptr else null,
        .preTransform = support.capabilities.currentTransform,
        .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = mode,
        .clipped = c.VK_TRUE,
        .oldSwapchain = @ptrCast(c.VK_NULL_HANDLE),
    };

    var swapchain: c.VkSwapchainKHR = undefined;
    if (c.vkCreateSwapchainKHR(self.device, &create_info, null, &swapchain) != c.VK_SUCCESS) {
        return error.VkCreateSwapchainFailed;
    }

    var real_image_count: u32 = 0;
    _ = c.vkGetSwapchainImagesKHR(self.device, swapchain, &real_image_count, null);
    const images = try gpa.alloc(c.VkImage, real_image_count);
    _ = c.vkGetSwapchainImagesKHR(self.device, swapchain, &real_image_count, images.ptr);
    const image_views = try self.createImageViews(gpa, images, format.format);

    return .{
        .swapchain = swapchain,
        .images = images,
        .format = format.format,
        .extent = extent,
        .image_views = image_views,
    };
}

pub fn deinit(self: *Swapchain) void {
    _ = c.vkDeviceWaitIdle(self.vulkan.device);

    for (self.views) |image_view| {
        c.vkDestroyImageView(self.device, image_view, null);
    }
    self.gpa.free(self.views);
    self.gpa.free(self.images);

    c.vkDestroySwapchainKHR(self.vulkan.device, self.swapchain, null);
}

fn chooseSwapExtent(
    capabilities: *const c.VkSurfaceCapabilitiesKHR,
    app: *const App,
) c.VkExtent2D {
    if (capabilities.currentExtent.width != std.math.maxInt(u32)) {
        return capabilities.currentExtent;
    } else {
        return .{
            .width = app.terminal.size.width,
            .height = app.terminal.size.height,
        };
    }
}

fn createImageViews(
    self: *Vulkan,
    gpa: Allocator,
    images: []const c.VkImage,
    format: c.VkFormat,
) ![]const c.VkImageView {
    const image_views = try gpa.alloc(c.VkImageView, images.len);
    for (images, 0..) |image, i| {
        const create_info: c.VkImageViewCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .image = image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
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

        if (c.vkCreateImageView(self.device, &create_info, null, &image_views[i]) != c.VK_SUCCESS) {
            return error.VkCreateImageViewFailed;
        }
    }

    return image_views;
}

fn createRenderPipeline(
    gpa: Allocator,
    device: c.VkDevice,
) !struct { layout: c.VkPipelineLayout, pipeline: c.VkPipeline, set_layout: c.VkDescriptorSetLayout } {
    const shader_code = @embedFile("comp.spv");
    const shader_buffer = try gpa.alignedAlloc(u8, @alignOf(u32), shader_code.len);
    defer gpa.free(shader_buffer);
    @memcpy(shader_buffer, shader_code);

    const shader_info: c.VkShaderModuleCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .codeSize = shader_buffer.len,
        .pCode = @ptrCast(@alignCast(shader_buffer.ptr)),
    };

    var shader_module: c.VkShaderModule = undefined;
    if (c.vkCreateShaderModule(device, &shader_info, null, &shader_module) != c.VK_SUCCESS) {
        return error.VkCreateShaderModuleFailed;
    }
    defer c.vkDestroyShaderModule(device, shader_module, null);

    const shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stage = c.VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = null,
    };

    comptime std.debug.assert(@sizeOf(App.Terminal) <= 128);
    const push_info: c.VkPushConstantRange = .{
        .offset = 0,
        .size = @sizeOf(App.Terminal),
        .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
    };

    // both of these are bound per window resize/monitor change (infrequently)
    // they will be allocated (one set per swapchain image) during swapchain
    // creation so they don't have to be re-bound per frame
    // writeonly storage image for writing to swapchain image
    const swapchain_image_binding: c.VkDescriptorSetLayoutBinding = .{
        .binding = 0,
        .descriptorCount = 1,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImmutableSamplers = null,
        .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
    };

    // readonly storage buffer for per-cell data (glyph cache and style cache indices)
    // split into glyph index binding and style index binding
    const cell_glyph_binding: c.VkDescriptorSetLayoutBinding = .{
        .binding = 1,
        .descriptorCount = 1,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImmutableSamplers = null,
        .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
    };

    const cell_style_binding: c.VkDescriptorSetLayoutBinding = .{
        .binding = 2,
        .descriptorCount = 1,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImmutableSamplers = null,
        .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
    };

    const bindings: []const c.VkDescriptorSetLayoutBinding = &.{
        swapchain_image_binding,
        cell_glyph_binding,
        cell_style_binding,
    };

    const set_layout_info: c.VkDescriptorSetLayoutCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .bindingCount = bindings.len,
        .pBindings = bindings.ptr,
    };

    var set_layout: c.VkDescriptorSetLayout = undefined;
    if (c.vkCreateDescriptorSetLayout(device, &set_layout_info, null, &set_layout) != c.VK_SUCCESS) {
        return error.VkCreateDescriptorSetLayoutFailed;
    }

    const layout_info: c.VkPipelineLayoutCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_info,
    };

    var layout: c.VkPipelineLayout = undefined;
    if (c.vkCreatePipelineLayout(device, &layout_info, null, &layout) != c.VK_SUCCESS) {
        return error.VkCreatePipelineLayoutFailed;
    }

    const pipeline_info: c.VkComputePipelineCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .layout = layout,
        .stage = shader_stage_info,
        .basePipelineHandle = @ptrCast(c.VK_NULL_HANDLE),
        .basePipelineIndex = -1,
    };

    var pipeline: c.VkPipeline = undefined;
    if (c.vkCreateComputePipelines(device, @ptrCast(c.VK_NULL_HANDLE), 1, &pipeline_info, null, &pipeline) != c.VK_SUCCESS) {
        return error.VkCreateComputePipelineFailed;
    }

    return .{
        .layout = layout,
        .pipeline = pipeline,
        .set_layout = set_layout,
    };
}

fn createGlobalSets(self: *Vulkan) ![]const c.VkDescriptorSet {
    const nsets: u32 = @intCast(self.swapchain.images.len);
    var sets = try self.gpa.alloc(c.VkDescriptorSet, nsets);

    const set_info: c.VkDescriptorSetAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = null,
        .descriptorPool = self.descriptor_pool,
        .descriptorSetCount = nsets,
        .pSetLayouts = &self.global_set_layout,
    };
    if (c.vkAllocateDescriptorSets(self.device, &set_info, sets.ptr) != c.VK_SUCCESS) {
        return error.VkAllocateDescriptorSetsFailed;
    }

    return sets;
}

fn bindGlobalSets(self: *Vulkan) !void {
    for (self.swapchain.image_views, 0..) |iv, i| {
        _ = iv;
        _ = i;
        //     const swapchain_image_set: c.VkWriteDescriptorSet = .{
        //         .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        //         .pNext = null,
        //         .dstSet = self.global_sets[i],
        //         .dstBinding = 0,
        //         .dstArrayElement = 0,
        //         .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        //         .descriptorCount = 1,
        //         .pBufferInfo = null,
        //         .pImageInfo = &c.VkDescriptorImageInfo{
        //             .sampler = @ptrCast(c.VK_NULL_HANDLE),
        //             .imageView = iv,
        //             .imageLayout = c.VK_IMAGE_LAYOUT_GENERAL,
        //         },
        //         .pTexelBufferView = null,
        //     };
        //
        //     const cells = &self.app.cells;
        //     const cell_glyph_set: c.VkWriteDescriptorSet = .{
        //         .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        //         .pNext = null,
        //         .dstSet = self.global_sets[i],
        //         .dstBinding = 0,
        //         .dstArrayElement = 0,
        //         .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        //         .descriptorCount = 1,
        //         .pBufferInfo = &c.VkDescriptorBufferInfo{
        //             .buffer = cells.items(.glyph).ptr,
        //             .offset = 0,
        //             .range = cells.len * @sizeOf(App.Cell.glyph),
        //         },
        //         .pImageInfo = null,
        //         .pTexelBufferView = null,
        //     };
        //
        //     const cell_style_set: c.VkWriteDescriptorSet = .{
        //         .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        //         .pNext = null,
        //         .dstSet = self.global_sets[i],
        //         .dstBinding = 0,
        //         .dstArrayElement = 0,
        //         .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        //         .descriptorCount = 1,
        //         .pBufferInfo = &c.VkDescriptorBufferInfo{
        //             .buffer = cells.items(.style).ptr,
        //             .offset = 0,
        //             .range = cells.len * @sizeOf(App.Cell.style),
        //         },
        //         .pImageInfo = null,
        //         .pTexelBufferView = null,
        //     };
        //
        //     const sets: []const c.VkWriteDescriptorSet = &.{
        //         swapchain_image_set,
        //         cell_glyph_set,
        //         cell_style_set,
        //     };
        //
        //     c.vkUpdateDescriptorSets(self.device, @intCast(sets.len), sets.ptr, 0, null);
    }
}

fn createCommandPool(device: c.VkDevice, graphics_family: u32) !c.VkCommandPool {
    const create_info: c.VkCommandPoolCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = null,
        .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphics_family,
    };

    var command_pool: c.VkCommandPool = undefined;
    if (c.vkCreateCommandPool(device, &create_info, null, &command_pool) != c.VK_SUCCESS) {
        return error.VkCreateCommandPoolFailed;
    }

    return command_pool;
}

fn createCommandBuffer(self: *Vulkan) !c.VkCommandBuffer {
    const alloc_info: c.VkCommandBufferAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = null,
        .commandPool = self.command_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    var command_buffer: c.VkCommandBuffer = undefined;
    if (c.vkAllocateCommandBuffers(self.device, &alloc_info, &command_buffer) != c.VK_SUCCESS) {
        return error.VkAllocateCommandBuffersFailed;
    }

    return command_buffer;
}

fn recordCommandBuffer(
    self: *Vulkan,
    command_buffer: c.VkCommandBuffer,
    image_index: u32,
) !void {
    // render command buffer:
    // * bind compute pipeline
    // * bind cell SSBO and swapchain image
    // * push constants for general terminal information
    // * image memory barrier: undefined -> general for compute
    // * dispatch compute
    // * image memory barrier: general -> present_src for presentation
    const begin_info: c.VkCommandBufferBeginInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = null,
        .flags = 0,
        .pInheritanceInfo = null,
    };
    if (c.vkBeginCommandBuffer(command_buffer, &begin_info) != c.VK_SUCCESS) {
        return error.VkBeginCommandBufferFailed;
    }

    c.vkCmdBindPipeline(command_buffer, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);

    // const atlas_image = try self.createGlyphAtlas();
    // TODO: move
    // const descriptor_dest1: c.VkWriteDescriptorSet = .{
    //     .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    //     .pNext = null,
    //     .dstSet = self.descriptor_set,
    //     .dstBinding = 0,
    //     .dstArrayElement = 0,
    //     .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    //     .descriptorCount = 1,
    //     .pBufferInfo = &c.VkDescriptorBufferInfo{
    //         .buffer = atlas_image,
    //         .offset = 0,
    //         .range = 7168,
    //     },
    //     .pImageInfo = null,
    //     .pTexelBufferView = null,
    // };
    // const descriptor_dest2: c.VkWriteDescriptorSet = .{
    //     .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    //     .pNext = null,
    //     .dstSet = self.descriptor_set,
    //     .dstBinding = 0,
    //     .dstArrayElement = 0,
    //     .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    //     .descriptorCount = 1,
    //     .pBufferInfo = null,
    //     .pImageInfo = &c.VkDescriptorImageInfo{
    //         .sampler = @ptrCast(c.VK_NULL_HANDLE),
    //         .imageView = self.swapchain.image_views[image_index],
    //         .imageLayout = c.VK_IMAGE_LAYOUT_GENERAL,
    //     },
    //     .pTexelBufferView = null,
    // };
    // const sets: []const c.VkWriteDescriptorSet = &.{ descriptor_dest1, descriptor_dest2 };
    // const sets: []const c.VkWriteDescriptorSet = &.{descriptor_dest2};
    // c.vkUpdateDescriptorSets(self.device, @intCast(sets.len), sets.ptr, 0, null);

    c.vkCmdBindDescriptorSets(command_buffer, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, &self.global_sets[image_index], 0, 0);

    c.vkCmdPushConstants(command_buffer, self.pipeline_layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(App.Terminal), &self.app.terminal);

    c.vkCmdPipelineBarrier(
        command_buffer,
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        c.VK_DEPENDENCY_BY_REGION_BIT, // TODO: correct?
        0,
        null,
        0,
        null,
        1,
        &@as(c.VkImageMemoryBarrier, .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = 0,
            .dstAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = c.VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = self.queue_families.compute,
            .dstQueueFamilyIndex = self.queue_families.compute,
            .image = self.swapchain.images[image_index],
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        }),
    );

    const width = self.app.terminal.size.width;
    const height = self.app.terminal.size.height;
    c.vkCmdDispatch(command_buffer, (width + 7) / 8, (height + 7) / 8, 1);

    c.vkCmdPipelineBarrier(
        command_buffer,
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        c.VK_DEPENDENCY_BY_REGION_BIT, // TODO: correct?
        0,
        null,
        0,
        null,
        1,
        &@as(c.VkImageMemoryBarrier, .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = 0,
            .oldLayout = c.VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = self.queue_families.compute,
            .dstQueueFamilyIndex = self.queue_families.compute,
            .image = self.swapchain.images[image_index],
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        }),
    );

    if (c.vkEndCommandBuffer(command_buffer) != c.VK_SUCCESS) {
        return error.VkCommandBufferRecordFailed;
    }
}

pub fn drawFrame(self: *Vulkan) !void {
    _ = c.vkWaitForFences(self.device, 1, &self.sync_objects.in_flight, c.VK_TRUE, std.math.maxInt(u64));
    _ = c.vkResetFences(self.device, 1, &self.sync_objects.in_flight);

    var image_index: u32 = 0;
    const result = c.vkAcquireNextImageKHR(
        self.device,
        self.swapchain.swapchain,
        std.math.maxInt(u64),
        self.sync_objects.image_available,
        @ptrCast(c.VK_NULL_HANDLE),
        &image_index,
    );
    // TODO: other return values
    if (result == c.VK_ERROR_OUT_OF_DATE_KHR) {
        self.deinitBufferObjects();
        try self.initBufferObjects();
    }

    _ = c.vkResetCommandBuffer(self.command_buffer, 0);
    // TODO: cache this?
    try self.recordCommandBuffer(
        self.command_buffer,
        image_index,
    );

    const wait_semaphores: []const c.VkSemaphore = &.{self.sync_objects.image_available};
    const wait_stages: []const c.VkPipelineStageFlags = &.{
        // c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    };
    const signal_semaphores: []const c.VkSemaphore = &.{self.sync_objects.render_finished};
    const submit_info: c.VkSubmitInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = null,
        .waitSemaphoreCount = @intCast(wait_semaphores.len),
        .pWaitSemaphores = wait_semaphores.ptr,
        .pWaitDstStageMask = wait_stages.ptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &self.command_buffer,
        .signalSemaphoreCount = @intCast(signal_semaphores.len),
        .pSignalSemaphores = signal_semaphores.ptr,
    };

    // if (c.vkQueueSubmit(self.graphics_queue, 1, &submit_info, self.sync_objects.in_flight) != c.VK_SUCCESS) {
    if (c.vkQueueSubmit(self.compute_queue, 1, &submit_info, self.sync_objects.in_flight) != c.VK_SUCCESS) {
        return error.VkSubmitDrawFailed;
    }

    const present_info: c.VkPresentInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = null,
        .waitSemaphoreCount = @intCast(signal_semaphores.len),
        .pWaitSemaphores = signal_semaphores.ptr,
        .swapchainCount = 1,
        .pSwapchains = &self.swapchain.swapchain,
        .pImageIndices = &image_index,
        .pResults = null,
    };
    _ = c.vkQueuePresentKHR(self.presentation_queue, &present_info);
}

pub fn createBuffer(
    self: *Vulkan,
    size: c.VkDeviceSize,
    usage: c.VkBufferUsageFlags,
    properties: c.VkMemoryPropertyFlags,
    buffer: *c.VkBuffer,
    memory: *c.VkDeviceMemory,
) !void {
    const buffer_info: c.VkBufferCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .size = size,
        .usage = usage,
        .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,
    };

    if (c.vkCreateBuffer(self.device, &buffer_info, null, buffer) != c.VK_SUCCESS) {
        return error.VkCreateBufferFailed;
    }

    var requirements: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(self.device, buffer.*, &requirements);

    const alloc_info: c.VkMemoryAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = null,
        .allocationSize = requirements.size,
        .memoryTypeIndex = findMemoryType(self.physical_device, requirements.memoryTypeBits, properties),
    };

    if (c.vkAllocateMemory(self.device, &alloc_info, null, memory) != c.VK_SUCCESS) {
        return error.VkAllocateMemoryFailed;
    }

    _ = c.vkBindBufferMemory(self.device, buffer.*, memory.*, 0);
}

pub fn createCellAttributesBuffer(self: *Vulkan, cells: []const Cell) !c.VkBuffer {
    const size = @sizeOf(Cell) * cells.len;
    var usage: c.VkBufferUsageFlags = c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    var properties: c.VkMemoryPropertyFlags = c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    var staging_buffer: c.VkBuffer = undefined;
    var staging_memory: c.VkDeviceMemory = undefined;
    try self.createBuffer(size, usage, properties, &staging_buffer, &staging_memory);
    defer c.vkDestroyBuffer(self.device, staging_buffer, null);
    defer c.vkFreeMemory(self.device, staging_memory, null);

    var data: []Cell = undefined;
    data.len = cells.len;
    _ = c.vkMapMemory(self.device, staging_memory, 0, size, 0, @ptrCast(&data.ptr));
    @memcpy(data, cells);
    c.vkUnmapMemory(self.device, staging_memory);

    usage = c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    properties = c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    var attribute_buffer: c.VkBuffer = undefined;
    var attribute_memory: c.VkDeviceMemory = undefined;
    try self.createBuffer(size, usage, properties, &attribute_buffer, &attribute_memory);
    self.copyBuffer(staging_buffer, attribute_buffer, size);

    return attribute_buffer;
}

fn copyBuffer(self: *Vulkan, src: c.VkBuffer, dest: c.VkBuffer, size: c.VkDeviceSize) void {
    const alloc_info: c.VkCommandBufferAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = null,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool = self.command_pool,
        .commandBufferCount = 1,
    };

    var command_buffer: c.VkCommandBuffer = undefined;
    _ = c.vkAllocateCommandBuffers(self.device, &alloc_info, &command_buffer);
    defer c.vkFreeCommandBuffers(self.device, self.command_pool, 1, &command_buffer);

    const begin_info: c.VkCommandBufferBeginInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = null,
        .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = null,
    };
    _ = c.vkBeginCommandBuffer(command_buffer, &begin_info);

    const region: c.VkBufferCopy = .{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = size,
    };
    c.vkCmdCopyBuffer(command_buffer, src, dest, 1, &region);
    _ = c.vkEndCommandBuffer(command_buffer);

    const submit_info: c.VkSubmitInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = null,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = null,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = null,
        .pWaitDstStageMask = null,
    };
    _ = c.vkQueueSubmit(self.graphics_queue, 1, &submit_info, @ptrCast(c.VK_NULL_HANDLE));
    _ = c.vkQueueWaitIdle(self.graphics_queue);
}

fn findMemoryType(physical_device: c.VkPhysicalDevice, filter: u32, properties: c.VkMemoryPropertyFlags) u32 {
    var mem_properties: c.VkPhysicalDeviceMemoryProperties = undefined;
    c.vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    var i: u5 = 0;
    while (i < mem_properties.memoryTypeCount) : (i += 1) {
        if ((filter & (@as(u32, 1) << i) > 0) and (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    // TODO: what happens if this fails
    unreachable;
}
