const std = @import("std");
const Allocator = std.mem.Allocator;

const GlyphCache = @This();

const c = @cImport({
    @cInclude("locale.h");
    @cInclude("fontconfig/fontconfig.h");
    @cInclude("freetype2/ft2build.h");
    @cInclude("freetype/freetype.h");
});

const ASCII_COUNT: usize = 95;
const DYNAMIC_COUNT: usize = 256;
const ASCII_BEGIN: u8 = 32;
const ASCII_END: u8 = 126;

ft: c.FT_Library,
face: c.FT_Face,
mono_width: u32,
mono_height: u32,

// fixed size CPU-side glyph cache for caching render output of freetype
// ascii values are always kept around, while other unicode glyphs
// are cached in LRU fashion
ascii: []u8, // 95 entries (ascii printable characters 32 - 126)
dyn: []u8, // 256 entries
lru: []LinkedNode, // linked list
tail: u8,

const LinkedNode = struct {
    slot: u8,
    next: u8,
};

pub fn init(gpa: Allocator) !GlyphCache {
    const fstring = "Iosevka Nerd Font SemiBold:size=14";
    const config = c.FcInitLoadConfigAndFonts();
    const parse = c.FcNameParse(fstring) orelse return error.FontParseFailed;
    defer c.FcPatternDestroy(parse);

    if (c.FcConfigSubstitute(config, parse, c.FcMatchPattern) != c.FcTrue) return error.FcSubstituteFailed;
    c.FcDefaultSubstitute(parse);
    const fset = c.FcFontSetCreate() orelse return error.FcSetCreateFailed;
    defer c.FcFontSetDestroy(fset);
    const oset = c.FcObjectSetBuild(c.FC_FAMILY, c.FC_STYLE, c.FC_FILE, @as(?*u8, null));

    var result: c.FcResult = undefined;
    const patterns = c.FcFontSort(config, parse, c.FcTrue, 0, &result);
    if (patterns == null or patterns[0].nfont == 0) return error.NoFontsFound;
    defer c.FcFontSetSortDestroy(patterns);

    const pattern = c.FcFontRenderPrepare(config, parse, patterns[0].fonts[0]);
    if (pattern == null) return error.FontPrepareFailed;
    if (c.FcFontSetAdd(fset, pattern) != c.FcTrue) return error.FontSetAddFailed;

    const font = c.FcPatternFilter(fset[0].fonts[0], oset);
    defer c.FcPatternDestroy(font);

    var v: c.FcValue = undefined;
    if (c.FcPatternGet(font, c.FC_FILE, 0, &v) != c.FcResultMatch) return error.FcPatternFailed;
    const filepath: [*:0]const u8 = @ptrCast(v.u.f orelse return error.FcFilepathInvalid);

    var ft: c.FT_Library = undefined;
    if (c.FT_Init_FreeType(&ft) != 0) return error.FtInitFailed;

    var face: c.FT_Face = undefined;
    if (c.FT_New_Face(ft, filepath, 0, &face) != 0) return error.FaceInitFailed;
    if (c.FT_Set_Pixel_Sizes(face, 0, 14) != 0) return error.FaceInitFailed;

    const index = c.FT_Get_Char_Index(face, 'a');
    if (c.FT_Load_Glyph(face, index, c.FT_LOAD_DEFAULT) != 0) return error.LoadFailed;
    const glyph = &face[0].glyph[0];
    const advance_x: usize = @intCast(@divFloor(glyph.metrics.horiAdvance, 64));
    const advance_y: usize = @intCast(@divFloor(glyph.metrics.vertAdvance, 64));
    const slot_size = advance_x * advance_y;

    const ascii = try initAscii(gpa, face, slot_size);

    return .{
        .ft = ft,
        .face = face,
        .mono_width = @intCast(advance_x),
        .mono_height = @intCast(advance_y),
        .ascii = ascii,
        .dyn = try gpa.alloc(u8, slot_size * DYNAMIC_COUNT),
        .lru = try gpa.alloc(LinkedNode, DYNAMIC_COUNT),
        .tail = 0,
    };
}

pub fn request(self: *GlyphCache, cp: u32) ![]u8 {
    if (cp < 128) {
        // ascii
        std.debug.assert(cp > 31 and cp < 127);
        const stride = self.mono_width * self.mono_height;
        const slot = cp - ASCII_BEGIN;
        return self.ascii[slot * stride .. (slot + 1) * stride];
    } else {
        // unicode
        unreachable; // TODO
    }
}

fn initAscii(gpa: Allocator, face: c.FT_Face, slot_size: usize) ![]u8 {
    const buffer = try gpa.alloc(u8, slot_size * ASCII_COUNT);

    var char: u8 = ASCII_BEGIN;
    while (char <= ASCII_END) : (char += 1) {
        // load and render the glyph into the freetype glyph slot
        const index = c.FT_Get_Char_Index(face, char);
        if (c.FT_Load_Glyph(face, index, c.FT_LOAD_DEFAULT) != 0) return error.LoadFailed;
        const glyph = &face[0].glyph[0];
        const advance_x: usize = @intCast(@divFloor(glyph.metrics.horiAdvance, 64));
        const advance_y: usize = @intCast(@divFloor(glyph.metrics.vertAdvance, 64));
        std.debug.assert(advance_x * advance_y == slot_size);

        if (c.FT_Render_Glyph(glyph, c.FT_RENDER_MODE_NORMAL) != 0) return error.RenderFailed;

        // copy it into our buffer
        const slot_index = char - ASCII_BEGIN;
        const slot = buffer[slot_index * slot_size .. (slot_index + 1) * slot_size];
        var row: usize = 0;
        while (row < glyph.bitmap.rows) : (row += 1) {
            const pitch: usize = @intCast(glyph.bitmap.pitch);
            const width: usize = @intCast(glyph.bitmap.width);
            const in = glyph.bitmap.buffer[row * pitch .. (row + 1) * pitch];
            const out = slot[row * width .. (row + 1) * width];
            @memcpy(out, in);
        }
    }

    return buffer;
}

pub fn deinit(self: *GlyphCache, gpa: Allocator) void {
    gpa.free(self.ascii);
    gpa.free(self.dyn);
    gpa.free(self.lru);
    _ = c.FT_Done_Face(self.face);
    _ = c.FT_Done_FreeType(self.ft);
}
