import { sizeSnapshot } from 'rollup-plugin-size-snapshot'

import json from 'rollup-plugin-json'
import commonjs from 'rollup-plugin-commonjs'
import globals from 'rollup-plugin-node-globals'
import replace from 'rollup-plugin-replace'
import resolve from 'rollup-plugin-node-resolve'
import { terser } from 'rollup-plugin-terser'
import visualizer from 'rollup-plugin-visualizer'

const isProd = env => env === 'production'

const config = (env, format) => ({
  input: 'compiled/index.js',
  output: {
    file: `./build/tracker.${format}.${env}.js`,
    name: 'detect',
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
    globals(),
    isProd(env) && sizeSnapshot(),
    isProd(env) && terser(),
    visualizer(),
  ]
})

export default [
  config('development', 'esm'),
  config('development', 'cjs'),
  // config('production', 'esm'),
]
