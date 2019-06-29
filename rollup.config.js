import { sizeSnapshot } from 'rollup-plugin-size-snapshot'

const json = require('rollup-plugin-json')
const commonjs = require('rollup-plugin-commonjs')
const globals = require('rollup-plugin-node-globals')
const replace = require('rollup-plugin-replace')
const resolve = require('rollup-plugin-node-resolve')
const { uglify } = require('rollup-plugin-uglify')
const visualizer = require('rollup-plugin-visualizer')

const isProd = env => env === 'production'

const config = (env, format) => ({
  input: 'compiled/index.js',
  output: {
    file: `./build/clmtracker2.${env}.js`,
    name: 'CLM-N',
    exports: 'named',
    format,
  },
  plugins: [
    json(),
    resolve(),
    replace({
      exclude: 'node_modules/**',
      'process.env.NODE_ENV': `'${env}'`,
    }),
    commonjs({
      ignoreGlobal: false,
    }),
    babel({
      exclude: 'node_modules/**',
      runtimeHelpers: true,
    }),
    globals(),
    isProd(env) && sizeSnapshot(),
    isProd(env) && uglify(),
    visualizer(),
  ]
})

export default [
  config('development', 'cjs'),
  config('production', 'cjs'),
]
