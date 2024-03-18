#version 450

layout(location = 0) in uvec2 cell_location;
layout(location = 1) in uint cell_character;
layout(location = 0) out vec3 frag_color;

layout(push_constant) uniform constants {
  uvec2 size;
  uvec2 cells;
  uvec2 cell_size;
  uvec2 offset;
} terminal;

uvec2 cell_coords[6] = uvec2[](
    // upper left
    uvec2(0, 0),
    uvec2(0, 1),
    uvec2(1, 0),
    // lower right
    uvec2(1, 1),
    uvec2(1, 0),
    uvec2(0, 1)
);

vec3 colors[6] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    // map tris to size of cell
    const uvec2 tris = cell_coords[gl_VertexIndex] * terminal.cell_size;
    // move cell to correct location on screen
    const uvec2 coords = tris + terminal.cell_size * cell_location;
    // shift by global window offset (border/justification)
    const uvec2 offset = coords + terminal.offset;
    // scale to clip coordinates
    const vec2 half_size = vec2(terminal.size) / 2.0;
    const vec2 clip = (vec2(offset) - half_size) / half_size;
    gl_Position = vec4(clip, 0.0, 1.0);
    frag_color = colors[gl_VertexIndex];
}
