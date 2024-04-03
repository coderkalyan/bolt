const std = @import("std");
const Wayland = @import("Wayland.zig");
// const Vulkan = @import("Vulkan.zig");
const VulkanInstance = @import("vulkan/Instance.zig");
const Swapchain = @import("vulkan/Swapchain.zig");
const Atlas = @import("vulkan/Atlas.zig");
const GlyphCache = @import("GlyphCache.zig");
const Allocator = std.mem.Allocator;

const App = @This();

gpa: Allocator,
wayland: Wayland,
vk_instance: VulkanInstance,
vk_swapchain: Swapchain,
vk_atlas: Atlas,
glyph_cache: GlyphCache,
cells: std.MultiArrayList(Cell),

configured: bool,
running: bool,

// mirrored to GPU via push constant
terminal: Terminal,

pub const Terminal = struct {
    // these can be smaller if necessary but we have the space
    // and shrinking them doesn't help much and causes potential problems
    // with glsl (which likes everything to be 32 bits)

    // window extent (in pixels)
    size: struct { width: u32, height: u32 },
    // number of cells
    cells: struct { cols: u32, rows: u32 },
    // cell size (in pixels)
    cell_size: struct { width: u32, height: u32 },
    // offset to first cell corner
    offset: struct { x: u32, y: u32 },
};

pub const Cell = struct {
    glyph: u32, // index into the glyph cache
    style: u32, // index into the style cache
};

pub fn init(gpa: Allocator) !App {
    const app: App = .{
        .gpa = gpa,
        .wayland = try Wayland.init(),
        .glyph_cache = try GlyphCache.init(gpa),
        .vk_instance = undefined,
        .vk_swapchain = undefined,
        .vk_atlas = undefined,
        .configured = false,
        .running = false,
        .terminal = .{
            .size = .{
                .width = 0,
                .height = 0,
            },
            .cells = .{ .cols = 0, .rows = 0 },
            .cell_size = .{
                .width = 10,
                .height = 25,
            },
            .offset = .{ .x = 0, .y = 0 },
        },
        .cells = .{},
    };

    return app;
}

pub fn configure(app: *App) !void {
    try app.wayland.configureToplevel();
    app.vk_instance = try VulkanInstance.init(app);
    errdefer app.vk_instance.deinit();
    app.vk_swapchain = try Swapchain.init(app.gpa, &app.vk_instance);
    errdefer app.vk_swapchain.deinit();
    app.vk_atlas = try Atlas.init(app.gpa, &app.vk_instance);
    errdefer app.vk_atlas.deinit();

    // TODO: this is testing
    var cp: u32 = 32;
    while (cp <= 126) : (cp += 1) {
        const glyph_index = try app.vk_atlas.request(cp);
        std.debug.print("glyph index: {}\n", .{glyph_index});
    }
    try app.vk_atlas.commit();

    app.configured = true;
}

pub fn deinit(app: *App) void {
    app.cells.deinit(app.gpa);
    app.wayland.deinit();
    app.glyph_cache.deinit(app.gpa);
}

pub fn deconfigure(app: *App) void {
    app.vk_atlas.deinit();
    app.vk_swapchain.deinit();
    app.vk_instance.deinit();
}

pub fn reinitSwapchain(app: *App) !void {
    app.vk_swapchain.deinit();
    app.vk_swapchain = try Swapchain.init(app.gpa, &app.vk_instance);
}

pub fn configureTerminal(app: *App, width_hint: u32, height_hint: u32) !void {
    const width = if (width_hint > 0) width_hint else 640;
    const height = if (height_hint > 0) height_hint else 360;

    const cell_width = app.terminal.cell_size.width;
    const cell_height = app.terminal.cell_size.height;

    app.terminal = .{
        .size = .{ .width = width, .height = height },
        .cells = .{ .cols = width / cell_width, .rows = height / cell_height },
        .cell_size = .{ .width = cell_width, .height = cell_height },
        .offset = .{ .x = (width % cell_width) / 2, .y = (height % cell_height) / 2 },
    };

    // const lipsum = @embedFile("lipsum.txt");
    const cols = app.terminal.cells.cols;
    const rows = app.terminal.cells.rows;
    try app.cells.resize(app.gpa, cols * rows);

    var k: usize = 0;
    var row: u32 = 0;
    while (row < rows) : (row += 1) {
        var col: u32 = 0;
        while (col < cols) : (col += 1) {
            app.cells.set(k, .{
                .glyph = 0,
                .style = 0xffffffff,
            });
            k += 1;
        }
    }

    // recreate swapchain if running
    // if (app.running) {
    //     app.vk_swapchain.deinit();
    //     std.debug.print("device deinit: {?}\n", .{app.vk_instance.device});
    //     app.vk_swapchain = try Swapchain.init(app.gpa, &app.vk_instance);
    //     std.debug.print("device reinit: {?}\n", .{app.vk_instance.device});
    // }
    // std.debug.print("{} {} {} {} {} {}\n", .{ app.terminal.size.width, app.terminal.size.height, app.terminal.cell_size.width, app.terminal.cell_size.height, app.terminal.cells.cols, app.terminal.cells.rows });
}
