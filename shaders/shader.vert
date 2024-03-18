#version 450

layout(location = 0) in uvec2 cell_location;
layout(location = 1) in uint cell_character;
layout(location = 0) out vec3 frag_color;

vec2 cell_coords[6] = vec2[](
    // upper left
    vec2(-1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0),
    // lower right
    vec2(1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(-1.0, 1.0)
);

vec3 colors[6] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

// vec2 scale = vec2(213, 46);
vec2 scale = vec2(212, 46);
vec2 size = 1 / scale;

void main() {
    const vec2 scaled = cell_coords[gl_VertexIndex] / scale;
    const vec2 base = scaled - 1.0; // + (vec2(-1.0, -1.0) / scale);
    // const vec2 base = cell_coords[gl_VertexIndex] - vec2(-1.0, -1.0);
    const vec2 loc = base;
    gl_Position = vec4(loc, 0.0, 1.0);
    frag_color = colors[gl_VertexIndex];
}
