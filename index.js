if (process.env.NODE_ENV === 'production') {
  module.exports = require('./clmtracker2.production.js')
} else {
  module.exports = require('./clmtracker2.development.js')
}
