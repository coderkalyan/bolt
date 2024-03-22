const std = @import("std");
const builtin = @import("builtin");
const App = @import("../App.zig");
const Wayland = @import("../Wayland.zig");
const Allocator = std.mem.Allocator;

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
    @cInclude("vulkan/vulkan_wayland.h");
    @cInclude("vulkan/vk_enum_string_helper.h");
    @cInclude("pixman.h");
});

const Vulkan = @This();

const QueueFamilies = struct {
    graphics: u32,
    presentation: u32,
    compute: u32,
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

// gpa: Allocator,
app: *App,

instance: c.VkInstance,
surface: c.VkSurfaceKHR,
phydev: c.VkPhysicalDevice,
device: c.VkDevice,
comp_index: u32,
comp_queue: c.VkQueue,
pres_index: u32,
pres_queue: c.VkQueue,
cmd_pool: c.VkCommandPool,
cmd_buffer: c.VkCommandBuffer,
ds_pool: c.VkDescriptorPool,
ds_layout_swapchain: c.VkDescriptorSetLayout,
pipeline: c.VkPipeline,
pipeline_layout: c.VkPipelineLayout,

pub fn init(app: *App) !Vulkan {
    var arena_allocator = std.heap.ArenaAllocator.init(app.gpa);
    defer arena_allocator.deinit();
    const arena = arena_allocator.allocator();

    // initialize vulkan instance
    const application_info: c.VkApplicationInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .pApplicationName = "bolt",
        .applicationVersion = c.VK_MAKE_VERSION(0, 0, 1),
        .pEngineName = "bolt",
        .engineVersion = c.VK_MAKE_VERSION(0, 0, 1),
        .apiVersion = c.VK_API_VERSION_1_0,
    };

    const extensions: []const [*:0]const u8 = &.{
        c.VK_KHR_SURFACE_EXTENSION_NAME,
        c.VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
    };

    const debug_layers: []const [*:0]const u8 = &.{
        "VK_LAYER_KHRONOS_validation",
        // "VK_LAYER_LUNARG_standard_validation",
    };

    const instance_info: c.VkInstanceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pApplicationInfo = &application_info,
        .enabledLayerCount = if (builtin.mode == .Debug) @intCast(debug_layers.len) else 0,
        .ppEnabledLayerNames = debug_layers.ptr,
        .enabledExtensionCount = @intCast(extensions.len),
        .ppEnabledExtensionNames = extensions.ptr,
    };

    var instance: c.VkInstance = undefined;
    if (c.vkCreateInstance(&instance_info, null, &instance) != c.VK_SUCCESS) {
        return error.VkCreateInstanceFailed;
    }

    // bind wayland surface to vulkan surface
    const surface_info: c.VkWaylandSurfaceCreateInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
        .pNext = null,
        .flags = 0,
        .display = @ptrCast(app.wayland.display),
        .surface = @ptrCast(app.wayland.surface),
    };

    var surface: c.VkSurfaceKHR = undefined;
    if (c.vkCreateWaylandSurfaceKHR(instance, &surface_info, null, &surface) != c.VK_SUCCESS) {
        return error.VkCreateSurfaceFailed;
    }

    // select a physical device
    var phydev_count: u32 = 0;
    var result = c.vkEnumeratePhysicalDevices(instance, &phydev_count, null);
    if (result != c.VK_SUCCESS) return error.VkEnumeratePhysicalDevicesFailed;
    if (phydev_count == 0) return error.NoValidPhysicalDevice;

    const phydevs = try arena.alloc(c.VkPhysicalDevice, phydev_count);
    result = c.vkEnumeratePhysicalDevices(instance, &phydev_count, phydevs.ptr);
    if (result != c.VK_SUCCESS) return error.VkEnumeratePhysicalDevicesFailed;

    var phydev: c.VkPhysicalDevice = null;
    var comp_index: u32 = undefined;
    var pres_index: u32 = undefined;
    for (phydevs) |device| {
        var properties: c.VkPhysicalDeviceProperties = undefined;
        var features: c.VkPhysicalDeviceFeatures = undefined;
        c.vkGetPhysicalDeviceProperties(device, &properties);
        c.vkGetPhysicalDeviceFeatures(device, &features);

        const families = findQueueFamilies(arena, device, surface) catch |err| switch (err) {
            error.QueueFamilyNotFound => continue,
            else => return err,
        };

        const support = try querySwapchainSupport(arena, device, surface);
        if (support.formats.len == 0 or support.modes.len == 0) continue;
        if (phydev == null or properties.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            // always choose a compatible device, preferring integrated graphics
            phydev = device;
            comp_index = families.comp;
            pres_index = families.pres;
        }
    }

    // create logical device
    const families: []const u32 = &.{ comp_index, pres_index };
    const queue_infos = try arena.alloc(c.VkDeviceQueueCreateInfo, families.len);

    const priority: f32 = 1.0;
    var unique_queues: u32 = 0;
    for (families, 0..) |family, i| {
        if (std.mem.indexOfScalar(u32, families[0..i], family) != null) continue;
        queue_infos[unique_queues] = .{
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

    const device_extensions: []const [*:0]const u8 = &.{
        c.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    // TODO: compatibility checks for swapchain
    const device_info: c.VkDeviceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueCreateInfoCount = unique_queues,
        .pQueueCreateInfos = queue_infos.ptr,
        .pEnabledFeatures = &device_features,
        .enabledExtensionCount = device_extensions.len,
        .ppEnabledExtensionNames = device_extensions.ptr,
        .enabledLayerCount = if (builtin.mode == .Debug) @intCast(debug_layers.len) else 0,
        .ppEnabledLayerNames = debug_layers.ptr,
    };

    var device: c.VkDevice = undefined;
    if (c.vkCreateDevice(phydev, &device_info, null, &device) != c.VK_SUCCESS) {
        return error.VkCreateDeviceFailed;
    }

    // create command pool
    const cmd_pool_info: c.VkCommandPoolCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = null,
        .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = comp_index,
    };

    var cmd_pool: c.VkCommandPool = undefined;
    if (c.vkCreateCommandPool(device, &cmd_pool_info, null, &cmd_pool) != c.VK_SUCCESS) {
        return error.VkCreateCommandPoolFailed;
    }

    // create descriptor pool
    const ds_image_size: c.VkDescriptorPoolSize = .{
        .type = c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 16,
    };

    const ds_buffer_size: c.VkDescriptorPoolSize = .{
        .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 16,
    };

    const ds_pool_sizes: []const c.VkDescriptorPoolSize = &.{
        ds_image_size,
        ds_buffer_size,
    };

    const ds_pool_info: c.VkDescriptorPoolCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = null,
        .flags = c.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .poolSizeCount = ds_pool_sizes.len,
        .pPoolSizes = ds_pool_sizes.ptr,
        .maxSets = 4,
    };

    var ds_pool: c.VkDescriptorPool = undefined;
    if (c.vkCreateDescriptorPool(device, &ds_pool_info, null, &ds_pool) != c.VK_SUCCESS) {
        return error.VkCreateDescriptorPoolFailed;
    }

    // initialize descriptor set layouts
    // the swapchain is only bound per window resize/monitor change (infrequently)
    // writeonly storage image for writing to swapchain image
    const swapchain_image_binding: c.VkDescriptorSetLayoutBinding = .{
        .binding = 0,
        .descriptorCount = 1,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImmutableSamplers = null,
        .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
    };

    var ds_layout_info: c.VkDescriptorSetLayoutCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .bindingCount = 1,
        .pBindings = &swapchain_image_binding,
    };

    var ds_layout_swapchain: c.VkDescriptorSetLayout = undefined;
    if (c.vkCreateDescriptorSetLayout(device, &ds_layout_info, null, &ds_layout_swapchain) != c.VK_SUCCESS) {
        return error.VkCreateDescriptorSetLayoutFailed;
    }

    // initialize queues
    var comp_queue: c.VkQueue = undefined;
    var pres_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, comp_index, 0, &comp_queue);
    c.vkGetDeviceQueue(device, pres_index, 0, &pres_queue);

    // create pipeline
    const shader_code = @embedFile("../comp.spv");
    const shader_buffer = try arena.alignedAlloc(u8, @alignOf(u32), shader_code.len);
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

    const layout_info: c.VkPipelineLayoutCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &ds_layout_swapchain,
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

    // create primary command buffer for rendering, this can be reused per frame
    // TODO: eventually, should try to cache the command buffer recording
    const cmd_buffer_info: c.VkCommandBufferAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = null,
        .commandPool = cmd_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    var cmd_buffer: c.VkCommandBuffer = undefined;
    if (c.vkAllocateCommandBuffers(device, &cmd_buffer_info, &cmd_buffer) != c.VK_SUCCESS) {
        return error.VkAllocateCommandBuffersFailed;
    }

    return .{
        .app = app,
        .instance = instance,
        .surface = surface,
        .phydev = phydev,
        .device = device,
        .comp_index = comp_index,
        .comp_queue = comp_queue,
        .pres_index = pres_index,
        .pres_queue = pres_queue,
        .cmd_pool = cmd_pool,
        .cmd_buffer = cmd_buffer,
        .ds_pool = ds_pool,
        .ds_layout_swapchain = ds_layout_swapchain,
        .pipeline = pipeline,
        .pipeline_layout = layout,
    };
}

pub fn deinit(self: *Vulkan) void {
    c.vkDestroyPipelineLayout(self.device, self.pipeline_layout, null);
    c.vkDestroyPipeline(self.device, self.pipeline, null);
    c.vkDestroyDescriptorSetLayout(self.device, self.ds_layout_swapchain, null);
    c.vkDestroyDescriptorPool(self.device, self.ds_pool, null);
    c.vkDestroyCommandPool(self.device, self.cmd_pool, null);
    c.vkDestroyDevice(self.device, null);
    c.vkDestroySurfaceKHR(self.instance, self.surface, null);
    c.vkDestroyInstance(self.instance, null);
}

fn findQueueFamilies(
    arena: Allocator,
    phydev: c.VkPhysicalDevice,
    surface: c.VkSurfaceKHR,
) !struct { comp: u32, pres: u32 } {
    var presentation: ?u32 = null;
    var compute: ?u32 = null;

    var family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(phydev, &family_count, null);
    const families = try arena.alloc(c.VkQueueFamilyProperties, family_count);
    c.vkGetPhysicalDeviceQueueFamilyProperties(phydev, &family_count, families.ptr);

    for (families, 0..) |family, i| {
        if ((family.queueFlags & c.VK_QUEUE_COMPUTE_BIT) != 0) {
            compute = @intCast(i);
        }

        var pres_support: c.VkBool32 = @intFromBool(false);
        const res = c.vkGetPhysicalDeviceSurfaceSupportKHR(
            phydev,
            @intCast(i),
            surface,
            &pres_support,
        );
        if (res != c.VK_SUCCESS) continue;
        if (pres_support == @intFromBool(true)) presentation = @intCast(i);
    }

    if (presentation == null) return error.QueueFamilyNotFound;
    if (compute == null) return error.QueueFamilyNotFound;

    return .{
        .comp = compute.?,
        .pres = presentation.?,
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
