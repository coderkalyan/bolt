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

instance: c.VkInstance,
surface: c.VkSurfaceKHR,
device: c.VkDevice,

// TODO: use zig allocator for vulkan api allocations
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
    const physical_device = try selectPhysicalDevice(instance, arena);
    const queue_families = try findQueueFamilies(physical_device, arena, surface);
    const device = try createLogicalDevice(arena, physical_device, queue_families, layers);

    var graphics_queue: c.VkQueue = undefined;
    var presentation_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, queue_families.graphics, 0, &graphics_queue);
    c.vkGetDeviceQueue(device, queue_families.presentation, 0, &presentation_queue);

    return .{
        .instance = instance,
        .surface = surface,
        .device = device,
    };
}

pub fn deinit(self: *Vulkan) void {
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

fn selectPhysicalDevice(instance: c.VkInstance, arena: Allocator) !c.VkPhysicalDevice {
    var device_count: u32 = 0;
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, null);
    if (device_count == 0) {
        return error.NoValidPhysicalDevice;
    }

    const devices = try arena.alloc(c.VkPhysicalDevice, device_count);
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr);

    // TODO: find lowest power device (i.e. integrated)
    return devices[0];
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

    return .{
        .graphics = graphics.?,
        .presentation = presentation.?,
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
