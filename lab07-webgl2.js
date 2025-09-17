/**
 * Esqueleto de um programa usando WegGL
 * Dessa vez usando as bibliotecas
 * macWebglUtils.js
 * MVnew.js do livro do Angel -- Interactive Computer Graphics
 */

"use strict";

// ==================================================================
// variáveis globais
var gl;
var gCanvas;
var gShader = {};

// Os códigos fonte dos shaders serão descritos em 
// strings para serem compilados e passados a GPU
var gsVertexShaderSrc;
var gsFragmentShaderSrc;

// Define o objeto a ser desenhado: uma lista de vértices
// com coordenadas no intervalo (0,0) a (200, 200)
var gaPositions = [
  // triangulo 1
  [50, 50],
  [150, 50],
  [100, 150],
  // triangulo 2
  [100, 150],
  [150, 50],
  [200, 150],
];

// ==================================================================
// chama a main quando terminar de carregar a janela
window.onload = main;

/**
 * programa principal.
 */
function main() {
  gCanvas = document.getElementById("glcanvas");
  gl = gCanvas.getContext('webgl2');
  if (!gl) alert("WebGL 2.0 isn't available");

  crieShaders();
  desenhe();
}

/**
 * cria e configura os shaders
 */
function crieShaders() {
  //  cria o programa
  gShader.program = makeProgram(gl, gsVertexShaderSrc, gsFragmentShaderSrc);
  gl.useProgram(gShader.program);

  // carrega dados na GPU
  var bufVertices = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bufVertices);
  gl.bufferData(gl.ARRAY_BUFFER, flatten(gaPositions), gl.STATIC_DRAW);
  // Associa as variáveis do shader ao buffer
  var aPositionLoc = gl.getAttribLocation(gShader.program, "aPosition");

  // Configuração do atributo para ler do buffer
  // atual ARRAY_BUFFER
  let size = 2;          // 2 elementos de cada vez - vec2
  let type = gl.FLOAT;   // tipo de 1 elemento = float 32 bits
  let normalize = false; // não normalize os dados
  let stride = 0;        // passo, quanto avançar a cada iteração depois de size*sizeof(type) 
  let offset = 0;        // começo do buffer
  gl.vertexAttribPointer(aPositionLoc, size, type, normalize, stride, offset);
  gl.enableVertexAttribArray(aPositionLoc);

  // resolve os uniforms
  gShader.uResolution = gl.getUniformLocation(gShader.program, "uResolution");
  gShader.uColor = gl.getUniformLocation(gShader.program, "uColor");

};

/**
 * Usa o shader para desenhar.
 * Assume que os dados já foram carregados e são estáticos.
 */
function desenhe() {

  // define como mapear coordenadas normalidas para o canvas
  gl.viewport(0, 0, gCanvas.width, gCanvas.height);
  // limpa o contexto
  gl.clearColor(0.0, 1.0, 1.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.uniform2f(gShader.uResolution, gCanvas.width, gCanvas.height);

  // desenhe 2 triangulos, cada um com uma cor aleatória
  for (let ii = 0; ii < 2; ii++) {
    // Set a random color.
    gl.uniform4f(gShader.uColor, Math.random(), Math.random(), Math.random(), 1);

    gl.drawArrays(gl.TRIANGLES, 3 * ii, 3);
  }
}

// ========================================================
// Código fonte dos shaders em GLSL
// a primeira linha deve conter "#version 300 es"
// para WebGL 2.0

gsVertexShaderSrc = `#version 300 es

// aPosition é um buffer de entrada
in vec2 aPosition;
uniform vec2 uResolution;

void main() {
    vec2 escala1 = aPosition / uResolution;
    vec2 escala2 = escala1 * 2.0;
    vec2 clipSpace = escala2 - 1.0;

    gl_Position = vec4(clipSpace, 0, 1);
}
`;

gsFragmentShaderSrc = `#version 300 es

// Vc deve definir a precisão do FS.
// Use highp ("high precision") para desktops e mediump para mobiles.
precision highp float;

// out define a saída 
out vec4 outColor;
uniform vec4 uColor;

void main() {
    outColor = uColor;
}
`;

