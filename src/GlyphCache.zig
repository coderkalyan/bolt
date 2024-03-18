const std = @import("std");

const GlyphCache = @This();

const c = @cImport({
    @cInclude("locale.h");
    @cInclude("fcft/fcft.h");
});

font: *c.fcft_font,
ascii: [128]*const c.fcft_glyph,

pub fn init() !GlyphCache {
    if (!c.fcft_init(c.FCFT_LOG_COLORIZE_AUTO, false, c.FCFT_LOG_CLASS_DEBUG)) {
        return error.FcftInitFailed;
    }

    _ = c.setlocale(c.LC_CTYPE, "en_US.UTF-8");
    const fonts: []const ?[*]const u8 = &.{"IosevkaCustom:size=13.5"};
    const attrs: ?[*]const u8 = null;
    const font = c.fcft_from_name(fonts.len, @constCast(fonts.ptr), attrs);
    if (font == null) {
        return error.FcftFontLoadFailed;
    }

    var i: u8 = 1;
    var ascii = [_]*const c.fcft_glyph{undefined} ** 128;
    // TODO: better solution to this, currently fill 0 with valid glyph (unused)
    ascii[0] = c.fcft_rasterize_char_utf32(font, 1, c.FCFT_SUBPIXEL_NONE);
    while (i < 128) : (i += 1) {
        ascii[i] = c.fcft_rasterize_char_utf32(font, i, c.FCFT_SUBPIXEL_NONE);
    }

    return .{
        .font = font.?,
        .ascii = ascii,
    };
}

pub fn deinit(self: *GlyphCache) void {
    c.fcft_destroy(self.font);
    c.fcft_fini();
}
