const fs = require('fs');
const path = require('path');

const src = path.join(__dirname, 'node_modules', '@vladmandic', 'face-api', 'model');
const dst = path.join(__dirname, 'public', 'models');

if (!fs.existsSync(dst)) fs.mkdirSync(dst, { recursive: true });

fs.readdirSync(src).forEach(f => {
  fs.copyFileSync(path.join(src, f), path.join(dst, f));
  console.log('copied', f);
});
console.log('done');
