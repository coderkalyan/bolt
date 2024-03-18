const std = @import("std");
const App = @import("main.zig").App;
const Wayland = @import("Wayland.zig");
const GlyphCache = @import("GlyphCache.zig");
const Allocator = std.mem.Allocator;

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
    @cInclude("vulkan/vulkan_wayland.h");
    @cInclude("vulkan/vk_enum_string_helper.h");
    @cInclude("pixman.h");
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

gpa: Allocator,
app: *App,
instance: c.VkInstance,
surface: c.VkSurfaceKHR,
physical_device: c.VkPhysicalDevice,
device: c.VkDevice,
queue_families: QueueFamilies,
graphics_queue: c.VkQueue,
presentation_queue: c.VkQueue,
command_pool: c.VkCommandPool,
sync_objects: SyncObjects,

swapchain: Swapchain,
framebuffers: []const c.VkFramebuffer,
render_pass: c.VkRenderPass,
pipeline_layout: c.VkPipelineLayout,
pipeline: c.VkPipeline,
command_buffer: c.VkCommandBuffer,

pub fn init(gpa: Allocator, app: *App, gc: *GlyphCache) !Vulkan {
    _ = gc;
    const app_info: c.VkApplicationInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .pApplicationName = "bolt",
        .applicationVersion = c.VK_MAKE_VERSION(0, 0, 1),
        .pEngineName = "No Engine",
        .engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = c.VK_API_VERSION_1_0,
    };

    const instance = try createInstance(&app_info);
    const surface = try createSurface(instance, app.wayland);
    const physical_device = try selectPhysicalDevice(gpa, instance, surface);
    const queue_families = try findQueueFamilies(gpa, physical_device, surface);
    const device = try createLogicalDevice(gpa, physical_device, queue_families);
    const command_pool = try createCommandPool(device, queue_families.graphics);
    const sync_objects = try createSyncObjects(device);

    var graphics_queue: c.VkQueue = undefined;
    var presentation_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, queue_families.graphics, 0, &graphics_queue);
    c.vkGetDeviceQueue(device, queue_families.presentation, 0, &presentation_queue);

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
        .command_pool = command_pool,
        .sync_objects = sync_objects,
        .swapchain = undefined,
        .framebuffers = &.{},
        .render_pass = null,
        .pipeline_layout = null,
        .pipeline = null,
        .command_buffer = null,
    };
    try vulkan.initBufferObjects();

    return vulkan;
}

pub fn deinit(self: *Vulkan) void {
    _ = c.vkDeviceWaitIdle(self.device);
    c.vkDestroySemaphore(self.device, self.sync_objects.image_available, null);
    c.vkDestroySemaphore(self.device, self.sync_objects.render_finished, null);
    c.vkDestroyFence(self.device, self.sync_objects.in_flight, null);
    c.vkDestroyCommandPool(self.device, self.command_pool, null);
    c.vkDestroyPipeline(self.device, self.pipeline, null);
    c.vkDestroyPipelineLayout(self.device, self.pipeline_layout, null);
    c.vkDestroyRenderPass(self.device, self.render_pass, null);
    self.deinitBufferObjects();
    c.vkDestroyDevice(self.device, null);
    c.vkDestroySurfaceKHR(self.instance, self.surface, null);
    c.vkDestroyInstance(self.instance, null);
}

pub fn initBufferObjects(self: *Vulkan) !void {
    self.swapchain = try self.createSwapchain();
    // TODO: small chance that the format changed and render pass
    // needs to be recreated
    if (self.render_pass == null) {
        self.render_pass = try self.createRenderPass();
        const pipeline_info = try self.createGraphicsPipeline();
        self.pipeline_layout = pipeline_info.layout;
        self.pipeline = pipeline_info.pipeline;
    }

    self.framebuffers = try self.createFramebuffers();
    self.command_buffer = try self.createCommandBuffer();
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
    gpa: Allocator,
    instance: c.VkInstance,
    surface: c.VkSurfaceKHR,
) !c.VkPhysicalDevice {
    var device_count: u32 = 0;
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, null);
    if (device_count == 0) {
        return error.NoValidPhysicalDevice;
    }

    const devices = try gpa.alloc(c.VkPhysicalDevice, device_count);
    defer gpa.free(devices);
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr);

    var chosen: c.VkPhysicalDevice = null;
    for (devices) |device| {
        var properties: c.VkPhysicalDeviceProperties = undefined;
        var features: c.VkPhysicalDeviceFeatures = undefined;
        _ = c.vkGetPhysicalDeviceProperties(device, &properties);
        _ = c.vkGetPhysicalDeviceFeatures(device, &features);

        if (features.geometryShader != @intFromBool(true)) continue;
        _ = findQueueFamilies(gpa, device, surface) catch |err| switch (err) {
            error.QueueFamilyNotFound => continue,
            else => return err,
        };
        // TODO: check if all required extensions are supported by device
        const swapchain_support = try querySwapchainSupport(gpa, device, surface);
        defer gpa.free(swapchain_support.formats);
        defer gpa.free(swapchain_support.modes);

        if (swapchain_support.formats.len == 0 or swapchain_support.modes.len == 0) continue;
        if (chosen == null or properties.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            // always choose a compatible device, preferring integrated graphics
            chosen = device;
        }
    }
    return chosen;
}

fn findQueueFamilies(
    gpa: Allocator,
    device: c.VkPhysicalDevice,
    surface: c.VkSurfaceKHR,
) !QueueFamilies {
    var graphics: ?u32 = null;
    var presentation: ?u32 = null;

    var family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, null);
    const families = try gpa.alloc(c.VkQueueFamilyProperties, family_count);
    defer gpa.free(families);
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

fn createLogicalDevice(
    gpa: Allocator,
    physical_device: c.VkPhysicalDevice,
    queue_families: QueueFamilies,
) !c.VkDevice {
    const layers: cstrings = &.{
        "VK_LAYER_KHRONOS_validation",
    };

    const families: []const u32 = &.{ queue_families.graphics, queue_families.presentation };
    const queue_create_infos = try gpa.alloc(c.VkDeviceQueueCreateInfo, families.len);
    defer gpa.free(queue_create_infos);

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

fn createSwapchain(self: *Vulkan) !Swapchain {
    const gpa = self.gpa;

    const support = try querySwapchainSupport(gpa, self.physical_device, self.surface);
    defer gpa.free(support.formats);
    defer gpa.free(support.modes);

    const format = format: {
        for (support.formats) |format| {
            if (format.format == c.VK_FORMAT_B8G8R8A8_SRGB and format.colorSpace == c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
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

pub fn deinitBufferObjects(self: *Vulkan) void {
    _ = c.vkDeviceWaitIdle(self.device);

    for (self.framebuffers) |framebuffer| {
        c.vkDestroyFramebuffer(self.device, framebuffer, null);
    }
    self.gpa.free(self.framebuffers);
    for (self.swapchain.image_views) |image_view| {
        c.vkDestroyImageView(self.device, image_view, null);
    }
    self.gpa.free(self.swapchain.image_views);
    self.gpa.free(self.swapchain.images);

    c.vkDestroySwapchainKHR(self.device, self.swapchain.swapchain, null);
}

fn chooseSwapExtent(
    capabilities: *const c.VkSurfaceCapabilitiesKHR,
    app: *const App,
) c.VkExtent2D {
    if (capabilities.currentExtent.width != std.math.maxInt(u32)) {
        return capabilities.currentExtent;
    } else {
        return .{
            .width = app.width,
            .height = app.height,
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

fn createRenderPass(self: *Vulkan) !c.VkRenderPass {
    const color_attachment: c.VkAttachmentDescription = .{
        .flags = 0,
        .format = self.swapchain.format,
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
    if (c.vkCreateRenderPass(self.device, &create_info, null, &render_pass) != c.VK_SUCCESS) {
        return error.VkCreateRenderPassFailed;
    }

    return render_pass;
}

fn createGraphicsPipeline(
    self: *Vulkan,
) !struct { layout: c.VkPipelineLayout, pipeline: c.VkPipeline } {
    const device = self.device;
    const render_pass = self.render_pass;

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

    const binding_description = comptime Cell.bindingDescription();
    const attribute_descriptions = &(comptime Cell.attributeDescriptions());
    const vertex_input_info: c.VkPipelineVertexInputStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding_description,
        .vertexAttributeDescriptionCount = attribute_descriptions.len,
        .pVertexAttributeDescriptions = attribute_descriptions.ptr,
    };

    const input_assembly: c.VkPipelineInputAssemblyStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = c.VK_FALSE,
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

    const viewport_state: c.VkPipelineViewportStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = null,
        .scissorCount = 1,
        .pScissors = null,
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
        .frontFace = c.VK_FRONT_FACE_COUNTER_CLOCKWISE,
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
    self: *Vulkan,
) ![]const c.VkFramebuffer {
    const framebuffers = try self.gpa.alloc(c.VkFramebuffer, self.swapchain.image_views.len);
    for (self.swapchain.image_views, 0..) |image_view, i| {
        const create_info: c.VkFramebufferCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .renderPass = self.render_pass,
            .attachmentCount = 1,
            .pAttachments = &image_view,
            .width = self.swapchain.extent.width,
            .height = self.swapchain.extent.height,
            .layers = 1,
        };

        if (c.vkCreateFramebuffer(self.device, &create_info, null, &framebuffers[i]) != c.VK_SUCCESS) {
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
    vertex_buffers: []const c.VkBuffer,
) !void {
    const render_pass = self.render_pass;
    const framebuffers = self.framebuffers;
    const extent = self.swapchain.extent;
    const pipeline = self.pipeline;

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

    const offsets: []const c.VkDeviceSize = &.{0};
    c.vkCmdBindVertexBuffers(command_buffer, 0, @intCast(vertex_buffers.len), vertex_buffers.ptr, offsets.ptr);

    // TODO: find the number of instances properly
    c.vkCmdDraw(command_buffer, 6, 212 * 43, 0, 0);

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

pub fn drawFrame(self: *Vulkan, vertex_buffers: []const c.VkBuffer) !void {
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
        vertex_buffers,
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

pub const Cell = struct {
    // row, col
    location: [2]u32,
    character: u8,

    pub fn bindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(Cell),
            .inputRate = c.VK_VERTEX_INPUT_RATE_INSTANCE,
        };
    }

    pub fn attributeDescriptions() [2]c.VkVertexInputAttributeDescription {
        return [2]c.VkVertexInputAttributeDescription{
            .{
                .binding = 0,
                .location = 0,
                .format = c.VK_FORMAT_R32G32_UINT,
                .offset = @offsetOf(Cell, "location"),
            },
            .{
                .binding = 0,
                .location = 1,
                .format = c.VK_FORMAT_R8_UINT,
                .offset = @offsetOf(Cell, "character"),
            },
        };
    }
};

pub fn createCellAttributesBuffer(self: *Vulkan, cells: []const Cell) !c.VkBuffer {
    const buffer_info: c.VkBufferCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .size = @sizeOf(Cell) * cells.len,
        .usage = c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,
    };

    var buffer: c.VkBuffer = undefined;
    if (c.vkCreateBuffer(self.device, &buffer_info, null, &buffer) != c.VK_SUCCESS) {
        return error.VkCreateBufferFailed;
    }

    var requirements: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(self.device, buffer, &requirements);

    const alloc_info: c.VkMemoryAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = null,
        .allocationSize = requirements.size,
        .memoryTypeIndex = findMemoryType(self.physical_device, requirements.memoryTypeBits, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };

    var memory: c.VkDeviceMemory = undefined;
    if (c.vkAllocateMemory(self.device, &alloc_info, null, &memory) != c.VK_SUCCESS) {
        return error.VkAllocateMemoryFailed;
    }

    _ = c.vkBindBufferMemory(self.device, buffer, memory, 0);

    var data: []Cell = undefined;
    data.len = cells.len;
    _ = c.vkMapMemory(self.device, memory, 0, buffer_info.size, 0, @ptrCast(&data.ptr));
    @memcpy(data, cells);
    c.vkUnmapMemory(self.device, memory);

    return buffer;
}

fn createGlyphAtlas(gpa: Allocator, physical_device: c.VkPhysicalDevice, device: c.VkDevice, gc: *const GlyphCache) !void {
    // for (gc.ascii) |glyph| {
    //     std.debug.print("cp: {} cols: {} advance x: {} y: {}\n", .{ glyph.cp, glyph.cols, glyph.advance.x, glyph.advance.y });
    //     std.debug.print("format: {} {} {} {} {}\n", .{ c.pixman_image_get_format(@ptrCast(glyph.pix)), c.PIXMAN_a1, c.PIXMAN_a8, c.PIXMAN_x8r8g8b8, c.PIXMAN_a8r8g8b8 });
    // }
    // const glyph = gc.ascii['a'];
    // const advance = glyph.advance;
    // _ = advance;
    // const data: [*]u8 = @ptrCast(c.pixman_image_get_data(@ptrCast(glyph.pix)));
    // // _ = data;
    // const stdout = std.io.getStdOut().writer();
    // std.debug.print("{} {} {} {}\n", .{ glyph.width, glyph.height, glyph.x, glyph.y });
    // // var width: u32 = @intCast(c.pixman_image_get_width(@ptrCast(gc.ascii['T'].pix)));
    // // var height: u32 = @intCast(c.pixman_image_get_height(@ptrCast(gc.ascii['T'].pix)));
    // var width: u32 = @intCast(glyph.width);
    // var height: u32 = @intCast(glyph.height);
    // const stride: u32 = @intCast(c.pixman_image_get_stride(@ptrCast(glyph.pix)));
    // var k: usize = 0;
    // var i: usize = 0;
    // while (i < height) : (i += 1) {
    //     var j: usize = 0;
    //     while (j < width) : (j += 1) {
    //         k = (i * stride) + j;
    //         try stdout.print("\x1b[38;2;{};{};{}m██", .{ data[k], data[k], data[k] });
    //     }
    //     std.debug.print("\n", .{});
    // }
    const width: u32 = @intCast(gc.ascii['a'].width);
    const height: u32 = @intCast(gc.ascii['a'].height);
    const stride: u32 = @intCast(c.pixman_image_get_stride(@ptrCast(gc.ascii['a'].pix)));

    var atlas = try gpa.alloc(u8, width * 128 * height);
    const cp_stride = width * height;
    var cp: usize = 0;
    while (cp < 128) : (cp += 1) {
        const glyph = gc.ascii[cp];
        const data: [*]u8 = @ptrCast(c.pixman_image_get_data(@ptrCast(glyph.pix)));
        const x = cp_stride * cp;
        var y: usize = 0;
        while (y < height) : (y += 1) {
            @memcpy(atlas[x + y * width .. x + y * width + width], data[y * stride .. y * stride + width]);
        }
    }

    // TODO: use a staging buffer
    const buffer_info: c.VkBufferCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .size = atlas.len,
        .usage = c.VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | c.VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
        .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,
    };

    var atlas_buffer: c.VkBuffer = undefined;
    if (c.vkCreateBuffer(device, &buffer_info, null, &atlas_buffer) != c.VK_SUCCESS) {
        return error.VkCreateBufferFailed;
    }

    var mem_reqs: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(device, atlas_buffer, &mem_reqs);

    const alloc_info: c.VkMemoryAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = null,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = findMemoryType(physical_device, mem_reqs.memoryTypeBits, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };

    var atlas_memory: c.VkDeviceMemory = undefined;
    if (c.vkAllocateMemory(device, &alloc_info, null, &atlas_memory) != c.VK_SUCCESS) {
        return error.VkAllocateMemoryFailed;
    }

    _ = c.vkBindBufferMemory(device, atlas_buffer, atlas_memory, 0);

    var data: []u8 = undefined;
    data.len = atlas.len;
    _ = c.vkMapMemory(device, atlas_memory, 0, atlas.len, 0, @ptrCast(&data.ptr));
    @memcpy(data, atlas);
    c.vkUnmapMemory(device, atlas_memory);

    var atlas_image: c.VkImage = undefined;
    const image_info: c.VkImageCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = null,
        .imageType = c.VK_IMAGE_TYPE_2D,
        .extent = .{ .width = width * 128, .height = height, .depth = 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .format = c.VK_FORMAT_R8_SRGB,
        .tiling = c.VK_IMAGE_TILING_OPTIMAL,
        .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        .usage = c.VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | c.VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | c.VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        .samples = c.VK_SAMPLE_COUNT_1_BIT,
        .flags = 0,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,
    };

    if (c.vkCreateImage(device, &image_info, null, &atlas_image) != c.VK_SUCCESS) {
        return error.VkCreateImageFailed;
    }

    c.vkGetImageMemoryRequirements(device, atlas_image, &mem_reqs);

    const image_alloc_info: c.VkMemoryAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = null,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = findMemoryType(physical_device, mem_reqs.memoryTypeBits, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };

    var image_memory: c.VkDeviceMemory = undefined;
    if (c.vkAllocateMemory(device, &image_alloc_info, null, &image_memory) != c.VK_SUCCESS) {
        return error.VkAllocateMemoryFailed;
    }

    // {
    //     var i: usize = 0;
    //     while (i < height) : (i += 1) {
    //         var j: usize = 0;
    //         while (j < width) : (j += 1) {
    //             const k = (i * width) + j + (cp_stride * 'M');
    //             std.debug.print("\x1b[38;2;{};{};{}m██", .{ atlas[k], atlas[k], atlas[k] });
    //         }
    //         std.debug.print("\n", .{});
    //     }
    // }
    // {
    //     var i: usize = 0;
    //     while (i < height) : (i += 1) {
    //         var j: usize = 0;
    //         while (j < width) : (j += 1) {
    //             const k = (i * width) + j + (cp_stride * 'a');
    //             std.debug.print("\x1b[38;2;{};{};{}m██", .{ atlas[k], atlas[k], atlas[k] });
    //         }
    //         std.debug.print("\n", .{});
    //     }
    // }
    // {
    //     var i: usize = 0;
    //     while (i < height) : (i += 1) {
    //         var j: usize = 0;
    //         while (j < width) : (j += 1) {
    //             const k = (i * width) + j + (cp_stride * 'x');
    //             std.debug.print("\x1b[38;2;{};{};{}m██", .{ atlas[k], atlas[k], atlas[k] });
    //         }
    //         std.debug.print("\n", .{});
    //     }
    // }
    // const image_size: c.VkDeviceSize = atlas.len;
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
