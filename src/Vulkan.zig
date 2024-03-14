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
    const queue_families = try findQueueFamilies(physical_device, arena);
    const device = try createLogicalDevice(physical_device, queue_families, layers);

    var graphics_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, queue_families.graphics, 0, &graphics_queue);

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
fn findQueueFamilies(device: c.VkPhysicalDevice, arena: Allocator) !QueueFamilies {
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

        // var presentation_support: c.VkBool32 = false;
        // c.vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, )
    }

    return .{
        .graphics = graphics.?,
        .presentation = presentation.?,
    };
}

fn createLogicalDevice(
    physical_device: c.VkPhysicalDevice,
    queue_families: QueueFamilies,
    layers: cstrings,
) !c.VkDevice {
    const queue_create_info: c.VkDeviceQueueCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueFamilyIndex = queue_families.graphics,
        .queueCount = 1,
        .pQueuePriorities = &@as(f32, 1.0),
    };

    var device_features: c.VkPhysicalDeviceFeatures = undefined;
    @memset(std.mem.asBytes(&device_features), 0);

    const device_create_info: c.VkDeviceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pQueueCreateInfos = &queue_create_info,
        .queueCreateInfoCount = 1,
        .pEnabledFeatures = &device_features,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
        .enabledLayerCount = @intCast(layers.len),
        .ppEnabledLayerNames = layers.ptr,
    };

    var device: c.VkDevice = undefined;
    if (c.vkCreateDevice(physical_device, &device_create_info, null, &device) != c.VK_SUCCESS) {
        return error.VkCreateDeviceFailed;
    }

    return device;
}
