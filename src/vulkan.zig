const std = @import("std");
const Allocator = std.mem.Allocator;

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
    @cInclude("vulkan/vulkan_wayland.h");
    @cInclude("vulkan/vk_enum_string_helper.h");
});

// TODO: use zig allocator
pub const Vulkan = struct {
    instance: c.VkInstance,

    pub fn init() !Vulkan {
        const instance = try initInstance();
        // const physical_device = try selectPhysicalDevice(instance);

        return .{
            .instance = instance,
        };
    }

    pub fn deinit(self: *Vulkan) void {
        c.vkDestroyInstance(self.instance, null);
    }

    fn initInstance() !c.VkInstance {
        const app_info: c.VkApplicationInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "bolt",
            .applicationVersion = c.VK_MAKE_VERSION(0, 0, 1),
            .pEngineName = "No Engine",
            .engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = c.VK_API_VERSION_1_0,
        };

        const extensions: []const [*]const u8 = &.{
            c.VK_KHR_SURFACE_EXTENSION_NAME,
            c.VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
        };
        const layers: []const [*]const u8 = &.{
            "VK_LAYER_KHRONOS_validation",
        };

        const create_info: c.VkInstanceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = layers.len,
            .ppEnabledLayerNames = layers.ptr,
            .enabledExtensionCount = extensions.len,
            .ppEnabledExtensionNames = extensions.ptr,
        };

        var instance: c.VkInstance = undefined;
        if (c.vkCreateInstance(&create_info, null, &instance) != c.VK_SUCCESS) {
            return error.VkCreateInstanceFailed;
        }

        return instance;
    }

    fn selectPhysicalDevice(instance: c.VkInstance, arena: Allocator) !c.VkPhysicalDevice {
        var device_count: u32 = 0;
        c.vkEnumeratePhysicalDevices(instance, &device_count, null);
        if (device_count == 0) {
            return error.NoPhysicalDevices;
        }

        const devices = arena.alloc(c.VkPhysicalDevice, device_count);
        c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr);

        var found = false;
        // TODO: find lowest power device (i.e. integrated)
        for (devices) |device| {
            found = true;
            // return
        }
    }
};
