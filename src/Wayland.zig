const std = @import("std");
const wayland = @import("wayland");
const Vulkan = @import("Vulkan.zig");
const App = @import("main.zig").App;
const wl = wayland.client.wl;
const xdg = wayland.client.xdg;

const orderZ = std.mem.orderZ;
var zero: u32 = 0;

pub const Wayland = @This();

app: *App,
display: *wl.Display,
registry: *wl.Registry,
compositor: *wl.Compositor,
surface: *wl.Surface,

wm_base: *xdg.WmBase,
xdg_surface: *xdg.Surface,
xdg_toplevel: *xdg.Toplevel,

pub fn init(app: *App) !Wayland {
    const display = try wl.Display.connect(null);
    const registry = try display.getRegistry();

    var bindings: Bindings = .{};
    registry.setListener(*@TypeOf(bindings), registryListener, &bindings);
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    if (bindings.compositor == null) return error.AcquisitionFailed;
    if (bindings.wm_base == null) return error.AcquisitionFailed;
    const compositor = bindings.compositor.?;
    const wm_base = bindings.wm_base.?;

    const surface = try compositor.createSurface();
    const xdg_surface = try wm_base.getXdgSurface(surface);
    const xdg_toplevel = try xdg_surface.getToplevel();

    const state: Wayland = .{
        .app = app,
        .display = display,
        .registry = registry,
        .compositor = compositor,
        .surface = surface,

        .wm_base = wm_base,
        .xdg_surface = xdg_surface,
        .xdg_toplevel = xdg_toplevel,
    };

    var configured: bool = false;
    state.xdg_surface.setListener(*bool, xdgSurfaceListener, &configured);
    state.xdg_toplevel.setTitle("bolt");
    state.xdg_toplevel.setAppId("bolt");
    state.xdg_toplevel.setListener(*App, xdgToplevelListener, app);

    state.surface.commit();
    while (state.display.dispatch() == .SUCCESS and !configured) {}

    return state;
}

pub fn deinit(state: *Wayland) void {
    state.xdg_toplevel.destroy();
    state.xdg_surface.destroy();
    state.surface.destroy();
    state.display.disconnect();
}

fn seatListener(seat: *wl.Seat, event: wl.Seat.Event, _: *u32) void {
    switch (event) {
        .capabilities => |caps| {
            // TODO: no pointer capability binding for now
            if (caps.capabilities.keyboard) {
                const keyboard = seat.getKeyboard() catch return;
                keyboard.setListener(*u32, keyboardListener, &zero);
            }
        },
        .name => {},
    }
}

fn keyboardListener(keyboard: *wl.Keyboard, event: wl.Keyboard.Event, _: *u32) void {
    _ = keyboard;
    switch (event) {
        .keymap => {},
        .enter => {},
        .leave => {},
        .key => |key| {
            _ = key;
            std.debug.print("key press event\n", .{});
        },
        .modifiers => |modifiers| {
            _ = modifiers;
            std.debug.print("modifier event\n", .{});
        },
        .repeat_info => {},
    }
}

fn wmBaseListener(wm_base: *xdg.WmBase, event: xdg.WmBase.Event, _: *u32) void {
    switch (event) {
        .ping => |ping| {
            wm_base.pong(ping.serial);
        },
    }
}

fn xdgSurfaceListener(
    xdg_surface: *xdg.Surface,
    event: xdg.Surface.Event,
    configured: *bool,
) void {
    switch (event) {
        .configure => |configure| {
            // TODO: when do we need to re-configure?
            xdg_surface.ackConfigure(configure.serial);
            configured.* = true;

            // if (configured) {
            //     state.surface.commit();
            // } else {
            //     state.configured = true;
            // }
        },
    }
}

fn xdgToplevelListener(
    _: *xdg.Toplevel,
    event: xdg.Toplevel.Event,
    app: *App,
) void {
    switch (event) {
        .configure => |configure| {
            app.width = @intCast(configure.width);
            app.height = @intCast(configure.height);

            if (app.running) {
                app.vulkan.deinitBufferObjects();
                app.vulkan.initBufferObjects() catch return;
            }
        },
        .close => {
            app.running = false;
        },
    }
}

const Bindings = struct {
    compositor: ?*wl.Compositor = null,
    wm_base: ?*xdg.WmBase = null,
};

fn registryListener(
    registry: *wl.Registry,
    event: wl.Registry.Event,
    bindings: *Bindings,
) void {
    switch (event) {
        .global => |global| {
            if (orderZ(u8, global.interface, wl.Seat.getInterface().name) == .eq) {
                const seat = registry.bind(global.name, wl.Seat, 1) catch return;
                seat.setListener(*u32, seatListener, &zero);
            } else if (orderZ(u8, global.interface, wl.Compositor.getInterface().name) == .eq) {
                bindings.compositor = registry.bind(global.name, wl.Compositor, 1) catch return;
            } else if (orderZ(u8, global.interface, xdg.WmBase.getInterface().name) == .eq) {
                const wm_base = registry.bind(global.name, xdg.WmBase, 1) catch return;
                wm_base.setListener(*u32, wmBaseListener, &zero);
                bindings.wm_base = wm_base;
            }
        },
        .global_remove => {},
    }
}
