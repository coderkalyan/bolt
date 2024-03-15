const std = @import("std");
const Wayland = @import("wayland.zig").Wayland;
const Allocator = std.mem.Allocator;

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
    @cInclude("vulkan/vulkan_wayland.h");
    @cInclude("vulkan/vk_enum_string_helper.h");
});

const Vulkan = @This();
const cstrings = []const [*:0]const u8;
const QueueFamilies = struct {
    graphics: u32,
    presentation: u32,
};
const SwapChainSupport = struct {
    capabilities: c.VkSurfaceCapabilitiesKHR,
    formats: []const c.VkSurfaceFormatKHR,
    modes: []const c.VkPresentModeKHR,
};
const Swapchain = struct {
    swapchain: c.VkSwapchainKHR,
    images: []const c.VkImage,
    format: c.VkFormat,
    extent: c.VkExtent2D,
    image_views: []const c.VkImageView,
};
const SyncObjects = struct {
    image_available: c.VkSemaphore,
    render_finished: c.VkSemaphore,
    in_flight: c.VkFence,
};

instance: c.VkInstance,
surface: c.VkSurfaceKHR,
device: c.VkDevice,
swapchain: Swapchain,
render_pass: c.VkRenderPass,
pipeline_layout: c.VkPipelineLayout,
pipeline: c.VkPipeline,
framebuffers: []const c.VkFramebuffer,
command_pool: c.VkCommandPool,
command_buffer: c.VkCommandBuffer,
sync_objects: SyncObjects,
graphics_queue: c.VkQueue,
presentation_queue: c.VkQueue,

// TODO: use zig allocator or VMA for vulkan api allocations
pub fn init(gpa: Allocator, display: *const Wayland) !Vulkan {
    var arena_allocator = std.heap.ArenaAllocator.init(gpa);
    const arena = arena_allocator.allocator();

    const app_info: c.VkApplicationInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .pApplicationName = "bolt",
        .applicationVersion = c.VK_MAKE_VERSION(0, 0, 1),
        .pEngineName = "No Engine",
        .engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = c.VK_API_VERSION_1_0,
    };
    const layers: cstrings = &.{
        "VK_LAYER_KHRONOS_validation",
    };

    const instance = try createInstance(&app_info);
    const surface = try createSurface(instance, display);
    const physical_device = try selectPhysicalDevice(arena, instance, surface);
    const queue_families = try findQueueFamilies(physical_device, arena, surface);
    const device = try createLogicalDevice(arena, physical_device, queue_families, layers);
    const swapchain = try createSwapchain(arena, physical_device, device, surface, queue_families);
    const render_pass = try createRenderPass(device, swapchain.format);
    const pipeline_info = try createGraphicsPipeline(device, swapchain.extent, render_pass);
    const framebuffers = try createFramebuffers(arena, device, swapchain.image_views, render_pass, swapchain.extent);
    const command_pool = try createCommandPool(device, queue_families.graphics);
    const command_buffer = try createCommandBuffer(device, command_pool);
    const sync_objects = try createSyncObjects(device);
    // try recordCommandBuffer(command_buffer, render_pass, framebuffers, extent, )

    var graphics_queue: c.VkQueue = undefined;
    var presentation_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, queue_families.graphics, 0, &graphics_queue);
    c.vkGetDeviceQueue(device, queue_families.presentation, 0, &presentation_queue);

    return .{
        .instance = instance,
        .surface = surface,
        .device = device,
        .swapchain = swapchain,
        .render_pass = render_pass,
        .pipeline_layout = pipeline_info.layout,
        .pipeline = pipeline_info.pipeline,
        .framebuffers = framebuffers,
        .command_pool = command_pool,
        .command_buffer = command_buffer,
        .graphics_queue = graphics_queue,
        .presentation_queue = presentation_queue,
        .sync_objects = sync_objects,
    };
}

pub fn deinit(self: *Vulkan) void {
    c.vkDestroyCommandPool(self.device, self.command_pool, null);
    for (self.framebuffers) |framebuffer| c.vkDestroyFramebuffer(self.device, framebuffer, null);
    c.vkDestroyPipeline(self.device, self.pipeline, null);
    c.vkDestroyPipelineLayout(self.device, self.pipeline_layout, null);
    for (self.swapchain.image_views) |image_view| c.vkDestroyImageView(self.device, image_view, null);
    c.vkDestroyRenderPass(self.device, self.render_pass, null);
    c.vkDestroySwapchainKHR(self.device, self.swapchain.swapchain, null);
    c.vkDestroyDevice(self.device, null);
    c.vkDestroySurfaceKHR(self.instance, self.surface, null);
    c.vkDestroyInstance(self.instance, null);
}

fn createInstance(
    app_info: *const c.VkApplicationInfo,
) !c.VkInstance {
    const extensions: cstrings = &.{
        c.VK_KHR_SURFACE_EXTENSION_NAME,
        c.VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
    };
    const layers: cstrings = &.{
        "VK_LAYER_KHRONOS_validation",
    };
    const create_info: c.VkInstanceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pApplicationInfo = app_info,
        .enabledLayerCount = @intCast(layers.len),
        .ppEnabledLayerNames = layers.ptr,
        .enabledExtensionCount = @intCast(extensions.len),
        .ppEnabledExtensionNames = extensions.ptr,
    };

    var instance: c.VkInstance = undefined;
    if (c.vkCreateInstance(&create_info, null, &instance) != c.VK_SUCCESS) {
        return error.VkCreateInstanceFailed;
    }

    return instance;
}

fn createSurface(instance: c.VkInstance, display: *const Wayland) !c.VkSurfaceKHR {
    const create_info: c.VkWaylandSurfaceCreateInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
        .pNext = null,
        .flags = 0,
        .display = @ptrCast(display.display),
        .surface = @ptrCast(display.surface),
    };

    var surface: c.VkSurfaceKHR = undefined;
    if (c.vkCreateWaylandSurfaceKHR(instance, &create_info, null, &surface) != c.VK_SUCCESS) {
        return error.VkCreateSurfaceFailed;
    }

    return surface;
}

fn selectPhysicalDevice(
    arena: Allocator,
    instance: c.VkInstance,
    surface: c.VkSurfaceKHR,
) !c.VkPhysicalDevice {
    var device_count: u32 = 0;
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, null);
    if (device_count == 0) {
        return error.NoValidPhysicalDevice;
    }

    const devices = try arena.alloc(c.VkPhysicalDevice, device_count);
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr);

    var chosen: c.VkPhysicalDevice = null;
    for (devices) |device| {
        var properties: c.VkPhysicalDeviceProperties = undefined;
        var features: c.VkPhysicalDeviceFeatures = undefined;
        _ = c.vkGetPhysicalDeviceProperties(device, &properties);
        _ = c.vkGetPhysicalDeviceFeatures(device, &features);

        if (features.geometryShader != @intFromBool(true)) continue;
        _ = findQueueFamilies(device, arena, surface) catch |err| switch (err) {
            error.QueueFamilyNotFound => continue,
            else => return err,
        };
        // TODO: check if all required extensions are supported by device
        const swapchain_support = try querySwapChainSupport(arena, device, surface);
        if (swapchain_support.formats.len == 0 or swapchain_support.modes.len == 0) continue;
        if (chosen == null or properties.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            // always choose a compatible device, preferring integrated graphics
            chosen = device;
        }
    }
    return chosen;
}

// TODO: support failing to find queue families
fn findQueueFamilies(
    device: c.VkPhysicalDevice,
    arena: Allocator,
    surface: c.VkSurfaceKHR,
) !QueueFamilies {
    var graphics: ?u32 = null;
    var presentation: ?u32 = 0; // TODO

    var family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, null);
    const families = try arena.alloc(c.VkQueueFamilyProperties, family_count);
    c.vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.ptr);

    for (families, 0..) |family, i| {
        if ((family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) != 0) {
            graphics = @intCast(i);
        }

        var presentation_support: c.VkBool32 = @intFromBool(false);
        _ = c.vkGetPhysicalDeviceSurfaceSupportKHR(device, @intCast(i), surface, &presentation_support);
        if (presentation_support == @intFromBool(true)) presentation = @intCast(i);
    }

    if (graphics == null) return error.QueueFamilyNotFound;
    if (presentation == null) return error.QueueFamilyNotFound;

    return .{
        .graphics = graphics.?,
        .presentation = presentation.?,
    };
}

fn querySwapChainSupport(
    arena: Allocator,
    device: c.VkPhysicalDevice,
    surface: c.VkSurfaceKHR,
) !SwapChainSupport {
    var capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
    _ = c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities);

    var format_count: u32 = 0;
    _ = c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, null);
    var formats: []c.VkSurfaceFormatKHR = &.{};
    if (format_count > 0) {
        formats = try arena.alloc(c.VkSurfaceFormatKHR, format_count);
        _ = c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, formats.ptr);
    }

    var mode_count: u32 = 0;
    _ = c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, null);
    var modes: []c.VkPresentModeKHR = &.{};
    if (mode_count > 0) {
        modes = try arena.alloc(c.VkPresentModeKHR, mode_count);
        _ = c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, modes.ptr);
    }

    return .{
        .capabilities = capabilities,
        .formats = formats,
        .modes = modes,
    };
}

fn createLogicalDevice(
    arena: Allocator,
    physical_device: c.VkPhysicalDevice,
    queue_families: QueueFamilies,
    layers: cstrings,
) !c.VkDevice {
    const families: []const u32 = &.{ queue_families.graphics, queue_families.presentation };
    const queue_create_infos = try arena.alloc(c.VkDeviceQueueCreateInfo, families.len);

    const priority: f32 = 1.0;
    var unique_queues: u32 = 0;
    for (families, 0..) |family, i| {
        if (std.mem.indexOfScalar(u32, families[0..i], family) != null) continue;
        queue_create_infos[unique_queues] = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
        unique_queues += 1;
    }

    var device_features: c.VkPhysicalDeviceFeatures = undefined;
    @memset(std.mem.asBytes(&device_features), 0);

    const extensions: cstrings = &.{
        c.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    // TODO: compatibility checks for swapchain
    const device_create_info: c.VkDeviceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueCreateInfoCount = unique_queues,
        .pQueueCreateInfos = queue_create_infos.ptr,
        .pEnabledFeatures = &device_features,
        .enabledExtensionCount = extensions.len,
        .ppEnabledExtensionNames = extensions.ptr,
        .enabledLayerCount = @intCast(layers.len),
        .ppEnabledLayerNames = layers.ptr,
    };

    var device: c.VkDevice = undefined;
    if (c.vkCreateDevice(physical_device, &device_create_info, null, &device) != c.VK_SUCCESS) {
        return error.VkCreateDeviceFailed;
    }

    return device;
}

fn createSwapchain(
    arena: Allocator,
    physical_device: c.VkPhysicalDevice,
    device: c.VkDevice,
    surface: c.VkSurfaceKHR,
    queue_families: QueueFamilies,
) !Swapchain {
    const support = try querySwapChainSupport(arena, physical_device, surface);
    const format = format: {
        for (support.formats) |format| {
            if (format.format == c.VK_FORMAT_B8G8R8A8_SRGB and format.colorSpace == c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                break :format format;
            }
        }
        break :format support.formats[0];
    };
    const mode = c.VK_PRESENT_MODE_FIFO_KHR; // guaranteed-exist and energy efficient
    const extent = chooseSwapExtent(&support.capabilities);
    const max_image_count = support.capabilities.maxImageCount;
    const image_count = @min(
        if (max_image_count == 0) std.math.maxInt(u32) else max_image_count,
        support.capabilities.minImageCount + 1,
    );

    const distinct = queue_families.graphics != queue_families.presentation;
    const queue_family_indices: []const u32 = &.{
        queue_families.graphics,
        queue_families.presentation,
    };
    const create_info: c.VkSwapchainCreateInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = null,
        .flags = 0,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = if (distinct) c.VK_SHARING_MODE_CONCURRENT else c.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = if (distinct) 2 else 0,
        .pQueueFamilyIndices = if (distinct) queue_family_indices.ptr else null,
        .preTransform = support.capabilities.currentTransform,
        .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = mode,
        .clipped = c.VK_TRUE,
        .oldSwapchain = @ptrCast(c.VK_NULL_HANDLE),
    };

    var swapchain: c.VkSwapchainKHR = undefined;
    if (c.vkCreateSwapchainKHR(device, &create_info, null, &swapchain) != c.VK_SUCCESS) {
        return error.VkCreateSwapchainFailed;
    }

    var real_image_count: u32 = 0;
    _ = c.vkGetSwapchainImagesKHR(device, swapchain, &real_image_count, null);
    const images = try arena.alloc(c.VkImage, real_image_count);
    _ = c.vkGetSwapchainImagesKHR(device, swapchain, &real_image_count, images.ptr);

    const image_views = try createImageViews(arena, device, images, format.format);

    return .{
        .swapchain = swapchain,
        .images = images,
        .format = format.format,
        .extent = extent,
        .image_views = image_views,
    };
}

fn chooseSwapExtent(capabilities: *const c.VkSurfaceCapabilitiesKHR) c.VkExtent2D {
    if (capabilities.currentExtent.width != std.math.maxInt(u32)) {
        std.debug.print("extent: {} {}\n", .{ capabilities.currentExtent.width, capabilities.currentExtent.height });
        return capabilities.currentExtent;
    } else {
        // TODO: implement this properly
        return .{
            .width = 480,
            .height = 1080,
        };
    }
}

fn createImageViews(
    arena: Allocator,
    device: c.VkDevice,
    images: []const c.VkImage,
    format: c.VkFormat,
) ![]const c.VkImageView {
    const image_views = try arena.alloc(c.VkImageView, images.len);
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

        if (c.vkCreateImageView(device, &create_info, null, &image_views[i]) != c.VK_SUCCESS) {
            return error.VkCreateImageViewFailed;
        }
    }

    return image_views;
}

fn createRenderPass(device: c.VkDevice, format: c.VkFormat) !c.VkRenderPass {
    const color_attachment: c.VkAttachmentDescription = .{
        .flags = 0,
        .format = format,
        .samples = c.VK_SAMPLE_COUNT_1_BIT,
        .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = c.VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    const color_attachment_ref: c.VkAttachmentReference = .{
        .attachment = 0,
        .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    const subpass: c.VkSubpassDescription = .{
        .flags = 0,
        .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
        .inputAttachmentCount = 0,
        .pInputAttachments = null,
        .pResolveAttachments = null,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = null,
        .pDepthStencilAttachment = null,
    };

    const dependency: c.VkSubpassDependency = .{
        .srcSubpass = c.VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0,
    };

    const create_info: c.VkRenderPassCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    var render_pass: c.VkRenderPass = undefined;
    if (c.vkCreateRenderPass(device, &create_info, null, &render_pass) != c.VK_SUCCESS) {
        return error.VkCreateRenderPassFailed;
    }

    return render_pass;
}

fn createGraphicsPipeline(
    device: c.VkDevice,
    extent: c.VkExtent2D,
    render_pass: c.VkRenderPass,
) !struct { layout: c.VkPipelineLayout, pipeline: c.VkPipeline } {
    // TODO: probably shouldn't do this
    const vert_shader_code align(4) = @embedFile("vert.spv").*;
    const frag_shader_code align(4) = @embedFile("frag.spv").*;

    const vert_shader_module = try createShaderModule(device, &vert_shader_code);
    const frag_shader_module = try createShaderModule(device, &frag_shader_code);
    defer c.vkDestroyShaderModule(device, vert_shader_module, null);
    defer c.vkDestroyShaderModule(device, frag_shader_module, null);

    const vert_shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stage = c.VK_SHADER_STAGE_VERTEX_BIT,
        .module = vert_shader_module,
        .pName = "main",
        .pSpecializationInfo = null,
    };
    const frag_shader_stage_info: c.VkPipelineShaderStageCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_shader_module,
        .pName = "main",
        .pSpecializationInfo = null,
    };
    const shader_stages: []const c.VkPipelineShaderStageCreateInfo = &.{
        vert_shader_stage_info,
        frag_shader_stage_info,
    };

    const dynamic_states: []const c.VkDynamicState = &.{
        c.VK_DYNAMIC_STATE_VIEWPORT,
        c.VK_DYNAMIC_STATE_SCISSOR,
    };
    const dynamic_state: c.VkPipelineDynamicStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .dynamicStateCount = dynamic_states.len,
        .pDynamicStates = dynamic_states.ptr,
    };

    const vertex_input_info: c.VkPipelineVertexInputStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = null,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = null,
    };

    const input_assembly: c.VkPipelineInputAssemblyStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = c.VK_FALSE,
    };

    const viewport: c.VkViewport = .{
        .x = 0,
        .y = 0,
        .width = @floatFromInt(extent.width),
        .height = @floatFromInt(extent.height),
        .minDepth = 0,
        .maxDepth = 1.0,
    };
    const scissor: c.VkRect2D = .{ .offset = .{ .x = 0, .y = 0 }, .extent = extent };

    // TODO: we probably don't need dynamic viewport state here
    const viewport_state: c.VkPipelineViewportStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    const rasterizer: c.VkPipelineRasterizationStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .depthClampEnable = c.VK_FALSE,
        .rasterizerDiscardEnable = c.VK_FALSE,
        .polygonMode = c.VK_POLYGON_MODE_FILL,
        .lineWidth = 1.0,
        .cullMode = c.VK_CULL_MODE_BACK_BIT,
        .frontFace = c.VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = c.VK_FALSE,
        .depthBiasConstantFactor = 0.0,
        .depthBiasClamp = 0.0,
        .depthBiasSlopeFactor = 0.0,
    };

    const multisampling: c.VkPipelineMultisampleStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .sampleShadingEnable = c.VK_FALSE,
        .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
        .minSampleShading = 1.0,
        .pSampleMask = null,
        .alphaToCoverageEnable = c.VK_FALSE,
        .alphaToOneEnable = c.VK_FALSE,
    };

    const color_blend_attachment: c.VkPipelineColorBlendAttachmentState = .{
        .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
        .blendEnable = c.VK_FALSE,
        .srcColorBlendFactor = c.VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = c.VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = c.VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = c.VK_BLEND_OP_ADD,
    };
    const color_blend: c.VkPipelineColorBlendStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .logicOpEnable = c.VK_FALSE,
        .logicOp = c.VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
        .blendConstants = .{ 0.0, 0.0, 0.0, 0.0 },
    };

    const pipeline_layout_info: c.VkPipelineLayoutCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = null,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = null,
    };

    var pipeline_layout: c.VkPipelineLayout = undefined;
    if (c.vkCreatePipelineLayout(device, &pipeline_layout_info, null, &pipeline_layout) != c.VK_SUCCESS) {
        return error.VkCreatePipelineLayoutFailed;
    }

    const pipeline_info: c.VkGraphicsPipelineCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stageCount = 2,
        .pStages = shader_stages.ptr,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = null,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic_state,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = @ptrCast(c.VK_NULL_HANDLE),
        .basePipelineIndex = -1,
        .pTessellationState = null,
    };

    var pipeline: c.VkPipeline = undefined;
    if (c.vkCreateGraphicsPipelines(device, @ptrCast(c.VK_NULL_HANDLE), 1, &pipeline_info, null, &pipeline) != c.VK_SUCCESS) {
        return error.VkCreateGraphicsPipelineFailed;
    }

    return .{
        .layout = pipeline_layout,
        .pipeline = pipeline,
    };
}

fn createShaderModule(device: c.VkDevice, code: []align(4) const u8) !c.VkShaderModule {
    const create_info: c.VkShaderModuleCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .codeSize = code.len,
        .pCode = @ptrCast(@alignCast(code.ptr)),
    };

    var module: c.VkShaderModule = undefined;
    if (c.vkCreateShaderModule(device, &create_info, null, &module) != c.VK_SUCCESS) {
        return error.VkCreateShaderModuleFailed;
    }

    return module;
}

fn createFramebuffers(
    arena: Allocator,
    device: c.VkDevice,
    image_views: []const c.VkImageView,
    render_pass: c.VkRenderPass,
    extent: c.VkExtent2D,
) ![]const c.VkFramebuffer {
    const framebuffers = try arena.alloc(c.VkFramebuffer, image_views.len);
    for (image_views, 0..) |image_view, i| {
        const create_info: c.VkFramebufferCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &image_view,
            .width = extent.width,
            .height = extent.height,
            .layers = 1,
        };

        if (c.vkCreateFramebuffer(device, &create_info, null, &framebuffers[i]) != c.VK_SUCCESS) {
            return error.VkCreateFramebufferFailed;
        }
    }

    return framebuffers;
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

fn createCommandBuffer(device: c.VkDevice, command_pool: c.VkCommandPool) !c.VkCommandBuffer {
    const alloc_info: c.VkCommandBufferAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = null,
        .commandPool = command_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    var command_buffer: c.VkCommandBuffer = undefined;
    if (c.vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) != c.VK_SUCCESS) {
        return error.VkAllocateCommandBuffersFailed;
    }

    return command_buffer;
}

fn recordCommandBuffer(
    command_buffer: c.VkCommandBuffer,
    render_pass: c.VkRenderPass,
    framebuffers: []const c.VkFramebuffer,
    extent: c.VkExtent2D,
    pipeline: c.VkPipeline,
    image_index: u32,
) !void {
    const begin_info: c.VkCommandBufferBeginInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = null,
        .flags = 0,
        .pInheritanceInfo = null,
    };

    if (c.vkBeginCommandBuffer(command_buffer, &begin_info) != c.VK_SUCCESS) {
        return error.VkBeginCommandBufferFailed;
    }

    const clear_color: c.VkClearValue = .{ .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } } };
    const render_pass_info: c.VkRenderPassBeginInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = null,
        .renderPass = render_pass,
        .framebuffer = framebuffers[image_index],
        .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = extent },
        .clearValueCount = 1,
        .pClearValues = &clear_color,
    };

    c.vkCmdBeginRenderPass(command_buffer, &render_pass_info, c.VK_SUBPASS_CONTENTS_INLINE);
    c.vkCmdBindPipeline(command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    const viewport: c.VkViewport = .{
        .x = 0.0,
        .y = 0.0,
        .width = @floatFromInt(extent.width),
        .height = @floatFromInt(extent.height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };
    c.vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    const scissor: c.VkRect2D = .{
        .offset = .{ .x = 0, .y = 0 },
        .extent = extent,
    };
    c.vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    c.vkCmdDraw(command_buffer, 3, 1, 0, 0);

    c.vkCmdEndRenderPass(command_buffer);
    if (c.vkEndCommandBuffer(command_buffer) != c.VK_SUCCESS) {
        return error.VkCommandBufferRecordFailed;
    }
}

fn createSyncObjects(device: c.VkDevice) !SyncObjects {
    var sync: SyncObjects = undefined;
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

    if (c.vkCreateSemaphore(device, &semaphore_info, null, &sync.image_available) != c.VK_SUCCESS) {
        return error.VkCreateSemaphoreFailed;
    }
    if (c.vkCreateSemaphore(device, &semaphore_info, null, &sync.render_finished) != c.VK_SUCCESS) {
        return error.VkCreateSemaphoreFailed;
    }
    if (c.vkCreateFence(device, &fence_info, null, &sync.in_flight) != c.VK_SUCCESS) {
        return error.VkCreateSemaphoreFailed;
    }

    return sync;
}

pub fn drawFrame(self: *Vulkan) !void {
    _ = c.vkWaitForFences(self.device, 1, &self.sync_objects.in_flight, c.VK_TRUE, std.math.maxInt(u64));
    _ = c.vkResetFences(self.device, 1, &self.sync_objects.in_flight);

    var image_index: u32 = 0;
    _ = c.vkAcquireNextImageKHR(
        self.device,
        self.swapchain.swapchain,
        std.math.maxInt(u64),
        self.sync_objects.image_available,
        @ptrCast(c.VK_NULL_HANDLE),
        &image_index,
    );

    _ = c.vkResetCommandBuffer(self.command_buffer, 0);
    // TODO: make this a "method" style
    try recordCommandBuffer(
        self.command_buffer,
        self.render_pass,
        self.framebuffers,
        self.swapchain.extent,
        self.pipeline,
        image_index,
    );

    const wait_semaphores: []const c.VkSemaphore = &.{self.sync_objects.image_available};
    const wait_stages: []const c.VkPipelineStageFlags = &.{
        c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
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

    if (c.vkQueueSubmit(self.graphics_queue, 1, &submit_info, self.sync_objects.in_flight) != c.VK_SUCCESS) {
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
