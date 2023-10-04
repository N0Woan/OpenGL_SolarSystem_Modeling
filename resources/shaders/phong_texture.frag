#version 330 core

precision mediump float;
in vec3 normal_interp;  // Surface normal
in vec3 vert_pos;       // Vertex position
in vec3 color_interp;
in vec2 texcoord_interp;

uniform mat3 K_materials;
uniform mat3 I_light;

uniform float phong_factor; // Shininess
uniform float shininess; // Shininess
uniform vec3 light_pos; // Light position
uniform int option;
out vec4 fragColor;


uniform sampler2D texture0;
uniform sampler2D texture1;
uniform sampler2D texture2;
uniform sampler2D texture3;
uniform sampler2D texture4;
uniform sampler2D texture5;
uniform sampler2D texture6;
uniform sampler2D texture7;
uniform sampler2D texture8;
uniform sampler2D texture9;
uniform sampler2D texture10;
uniform sampler2D texture11;

void main() {
  vec3 N = normalize(normal_interp);
  vec3 L = normalize(light_pos - vert_pos);
  vec3 R = reflect(-L, N);      // Reflected light vector
  vec3 V = normalize(-vert_pos); // Vector to viewer

  vec3 lv = light_pos - vert_pos;
  float lvd = 1.0/(dot(lv, lv));
  float specAngle = max(dot(R, V), 0.0);
  float specular = pow(specAngle, shininess);
  vec3 g = vec3(lvd*max(dot(L, N), 0.0), specular, 1.0);
  vec3 rgb = matrixCompMult(K_materials, I_light) * g; // +  colorInterp;

  fragColor = vec4(rgb, 1.0);
  vec4 color_interp4 = vec4(color_interp, 1.0);
  float color_factor = 0.2;
  float texture_factor = 1.0 - (color_factor + phong_factor);
  if (option == 0) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture0, texcoord_interp);
  }
  else if (option == 1) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture1, texcoord_interp);
  }
  else if (option == 2) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture2, texcoord_interp);
  }
  else if (option == 3) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture3, texcoord_interp);
  }
  else if (option == 4) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture4, texcoord_interp);
  }
  else if (option == 5) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture5, texcoord_interp);
  }
  else if (option == 6) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture6, texcoord_interp);
  }
  else if (option == 7) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture7, texcoord_interp);
  }
  else if (option == 8) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture8, texcoord_interp);
  }
  else if (option == 9) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture9, texcoord_interp);
  }
  else if (option == 10) {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture10, texcoord_interp);
  }
  else {
    fragColor = 0.0 * color_factor*color_interp4 + 0.0 * phong_factor*fragColor + texture_factor*texture(texture11, texcoord_interp);
  }
}