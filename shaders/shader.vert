#version 450

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
vec2 scale = vec2(213 / 46, 1) * 2;
void main() {
  gl_Position = vec4(cell_coords[gl_VertexIndex] / scale, 0.0, 1.0);
  frag_color = colors[gl_VertexIndex];
}
