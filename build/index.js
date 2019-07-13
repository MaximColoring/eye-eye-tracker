if (process.env.NODE_ENV === 'production') {
  module.exports = require('./tracker.cjs.production.js')
} else {
  module.exports = require('./tracker.cjs.development.js')
}
