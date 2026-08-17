/* ============================================================
   IMAGE PROCESSING ENGINE
   ============================================================ */
const clamp = v => v < 0 ? 0 : v > 255 ? 255 : v;

function toGray(imgData){
  const {data, width, height} = imgData;
  const out = new Float32Array(width*height);
  for(let i=0, p=0; i<data.length; i+=4, p++){
    out[p] = 0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2];
  }
  return out;
}

function grayToImageData(gray, width, height){
  const out = new ImageData(width, height);
  for(let p=0, i=0; p<gray.length; p++, i+=4){
    const v = clamp(gray[p]);
    out.data[i]=v; out.data[i+1]=v; out.data[i+2]=v; out.data[i+3]=255;
  }
  return out;
}

// separable box blur, per RGB channel
function boxBlur(imgData, radius){
  const {width, height} = imgData;
  const src = imgData.data;
  const tmp = new Float32Array(src.length);
  const out = new ImageData(width, height);
  const win = radius*2+1;

  // horizontal pass
  for(let y=0; y<height; y++){
    for(let c=0;c<3;c++){
      let sum=0;
      for(let x=-radius; x<=radius; x++){
        const xx = Math.min(width-1, Math.max(0,x));
        sum += src[(y*width+xx)*4+c];
      }
      for(let x=0; x<width; x++){
        tmp[(y*width+x)*4+c] = sum/win;
        const xOut = Math.min(width-1, x+radius+1);
        const xIn = Math.max(0, x-radius);
        sum += src[(y*width+xOut)*4+c] - src[(y*width+xIn)*4+c];
      }
    }
  }
  // vertical pass
  for(let x=0; x<width; x++){
    for(let c=0;c<3;c++){
      let sum=0;
      for(let y=-radius; y<=radius; y++){
        const yy = Math.min(height-1, Math.max(0,y));
        sum += tmp[(yy*width+x)*4+c];
      }
      for(let y=0; y<height; y++){
        out.data[(y*width+x)*4+c] = clamp(sum/win);
        const yOut = Math.min(height-1, y+radius+1);
        const yIn = Math.max(0, y-radius);
        sum += tmp[(yOut*width+x)*4+c] - tmp[(yIn*width+x)*4+c];
      }
    }
  }
  for(let p=0;p<width*height;p++) out.data[p*4+3]=255;
  return out;
}

function medianFilter(imgData, radius){
  const {width, height} = imgData;
  const src = imgData.data;
  const out = new ImageData(width, height);
  const winSize = (radius*2+1)*(radius*2+1);
  const buf = new Uint8ClampedArray(winSize);
  for(let y=0;y<height;y++){
    for(let x=0;x<width;x++){
      for(let c=0;c<3;c++){
        let n=0;
        for(let dy=-radius; dy<=radius; dy++){
          const yy = Math.min(height-1, Math.max(0, y+dy));
          for(let dx=-radius; dx<=radius; dx++){
            const xx = Math.min(width-1, Math.max(0, x+dx));
            buf[n++] = src[(yy*width+xx)*4+c];
          }
        }
        const sorted = Array.prototype.slice.call(buf,0,n).sort((a,b)=>a-b);
        out.data[(y*width+x)*4+c] = sorted[Math.floor(n/2)];
      }
      out.data[(y*width+x)*4+3]=255;
    }
  }
  return out;
}

function gaussianKernel1D(sigma){
  const radius = Math.max(1, Math.ceil(sigma*3));
  const size = radius*2+1;
  const k = new Float32Array(size);
  let sum=0;
  for(let i=-radius;i<=radius;i++){
    const v = Math.exp(-(i*i)/(2*sigma*sigma));
    k[i+radius]=v; sum+=v;
  }
  for(let i=0;i<size;i++) k[i]/=sum;
  return {k, radius};
}

function gaussianBlur(imgData, sigma){
  const {width, height} = imgData;
  const src = imgData.data;
  const {k, radius} = gaussianKernel1D(sigma);
  const tmp = new Float32Array(src.length);
  const out = new ImageData(width, height);

  for(let y=0;y<height;y++){
    for(let x=0;x<width;x++){
      for(let c=0;c<3;c++){
        let sum=0;
        for(let i=-radius;i<=radius;i++){
          const xx = Math.min(width-1, Math.max(0, x+i));
          sum += src[(y*width+xx)*4+c]*k[i+radius];
        }
        tmp[(y*width+x)*4+c]=sum;
      }
    }
  }
  for(let y=0;y<height;y++){
    for(let x=0;x<width;x++){
      for(let c=0;c<3;c++){
        let sum=0;
        for(let i=-radius;i<=radius;i++){
          const yy = Math.min(height-1, Math.max(0, y+i));
          sum += tmp[(yy*width+x)*4+c]*k[i+radius];
        }
        out.data[(y*width+x)*4+c]=clamp(sum);
      }
      out.data[(y*width+x)*4+3]=255;
    }
  }
  return out;
}

function convolveGray(gray, width, height, kernel){
  // kernel is 3x3 flat array
  const out = new Float32Array(width*height);
  for(let y=0;y<height;y++){
    for(let x=0;x<width;x++){
      let sum=0, ki=0;
      for(let dy=-1; dy<=1; dy++){
        const yy = Math.min(height-1, Math.max(0, y+dy));
        for(let dx=-1; dx<=1; dx++){
          const xx = Math.min(width-1, Math.max(0, x+dx));
          sum += gray[yy*width+xx]*kernel[ki++];
        }
      }
      out[y*width+x]=sum;
    }
  }
  return out;
}

const SOBEL_X = [-1,0,1,-2,0,2,-1,0,1];
const SOBEL_Y = [-1,-2,-1,0,0,0,1,2,1];
const PREWITT_X = [-1,0,1,-1,0,1,-1,0,1];
const PREWITT_Y = [-1,-1,-1,0,0,0,1,1,1];
const LAP_4 = [0,1,0,1,-4,1,0,1,0];
const LAP_8 = [1,1,1,1,-8,1,1,1,1];

function gradientMagnitude(gray, width, height, kx, ky){
  const gx = convolveGray(gray, width, height, kx);
  const gy = convolveGray(gray, width, height, ky);
  const mag = new Float32Array(width*height);
  let max=1;
  for(let i=0;i<mag.length;i++){
    mag[i]=Math.sqrt(gx[i]*gx[i]+gy[i]*gy[i]);
    if(mag[i]>max) max=mag[i];
  }
  return {mag, gx, gy, max};
}

function normalizeToImageData(arr, width, height, maxVal){
  const scale = maxVal ? 255/maxVal : 1;
  const gray = new Float32Array(arr.length);
  for(let i=0;i<arr.length;i++) gray[i]=arr[i]*scale;
  return grayToImageData(gray, width, height);
}

/* ---------------- Canny ---------------- */
function canny(imgData, sigma, low, high){
  const {width, height} = imgData;
  const gray = toGray(imgData);

  // blur (grayscale, single channel gaussian)
  const {k, radius} = gaussianKernel1D(sigma);
  const tmp = new Float32Array(width*height);
  const blurred = new Float32Array(width*height);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    let sum=0;
    for(let i=-radius;i<=radius;i++){
      const xx=Math.min(width-1,Math.max(0,x+i));
      sum+=gray[y*width+xx]*k[i+radius];
    }
    tmp[y*width+x]=sum;
  }
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    let sum=0;
    for(let i=-radius;i<=radius;i++){
      const yy=Math.min(height-1,Math.max(0,y+i));
      sum+=tmp[yy*width+x]*k[i+radius];
    }
    blurred[y*width+x]=sum;
  }

  const gx = convolveGray(blurred, width, height, SOBEL_X);
  const gy = convolveGray(blurred, width, height, SOBEL_Y);
  const mag = new Float32Array(width*height);
  const ang = new Float32Array(width*height);
  let maxMag=1;
  for(let i=0;i<mag.length;i++){
    mag[i]=Math.sqrt(gx[i]*gx[i]+gy[i]*gy[i]);
    let a = Math.atan2(gy[i], gx[i]) * 180/Math.PI;
    if(a<0) a+=180;
    ang[i]=a;
    if(mag[i]>maxMag) maxMag=mag[i];
  }

  // non-max suppression
  const nms = new Float32Array(width*height);
  for(let y=1;y<height-1;y++){
    for(let x=1;x<width-1;x++){
      const idx=y*width+x;
      const a = ang[idx];
      let n1,n2;
      if((a>=0&&a<22.5)||(a>=157.5&&a<=180)){ n1=mag[idx-1]; n2=mag[idx+1]; }
      else if(a>=22.5&&a<67.5){ n1=mag[idx-width+1]; n2=mag[idx+width-1]; }
      else if(a>=67.5&&a<112.5){ n1=mag[idx-width]; n2=mag[idx+width]; }
      else { n1=mag[idx-width-1]; n2=mag[idx+width+1]; }
      nms[idx] = (mag[idx]>=n1 && mag[idx]>=n2) ? mag[idx] : 0;
    }
  }

  // double threshold (low/high given 0-255, scale relative to observed max for robustness)
  const STRONG=255, WEAK=90, NONE=0;
  const cls = new Uint8ClampedArray(width*height);
  for(let i=0;i<nms.length;i++){
    if(nms[i]>=high) cls[i]=STRONG;
    else if(nms[i]>=low) cls[i]=WEAK;
    else cls[i]=NONE;
  }

  // hysteresis via BFS
  const result = new Uint8ClampedArray(width*height);
  const stack=[];
  for(let i=0;i<cls.length;i++){ if(cls[i]===STRONG){ result[i]=255; stack.push(i);} }
  while(stack.length){
    const idx = stack.pop();
    const x = idx%width, y=(idx/width)|0;
    for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
      if(dx===0&&dy===0) continue;
      const xx=x+dx, yy=y+dy;
      if(xx<0||yy<0||xx>=width||yy>=height) continue;
      const nIdx=yy*width+xx;
      if(cls[nIdx]===WEAK && result[nIdx]===0){ result[nIdx]=255; stack.push(nIdx); }
    }
  }

  return {
    blurred: grayToImageData(blurred, width, height),
    gradient: normalizeToImageData(mag, width, height, maxMag),
    nms: normalizeToImageData(nms, width, height, maxMag),
    edges: grayToImageData(Array.from(result), width, height)
  };
}

/* ============================================================
   SAMPLE IMAGE GENERATION
   ============================================================ */
function drawSample(ctx, w, h){
  const grad = ctx.createLinearGradient(0,0,w,h);
  grad.addColorStop(0,'#1c2620'); grad.addColorStop(1,'#3a4a3f');
  ctx.fillStyle = grad; ctx.fillRect(0,0,w,h);

  ctx.fillStyle = '#e8ede9';
  ctx.beginPath(); ctx.arc(w*0.32, h*0.36, Math.min(w,h)*0.16, 0, Math.PI*2); ctx.fill();

  ctx.fillStyle = '#7fffb0';
  ctx.fillRect(w*0.55, h*0.16, w*0.28, h*0.22);

  ctx.fillStyle = '#5cc8ff';
  ctx.beginPath();
  ctx.moveTo(w*0.22, h*0.86);
  ctx.lineTo(w*0.5, h*0.52);
  ctx.lineTo(w*0.78, h*0.86);
  ctx.closePath(); ctx.fill();

  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  ctx.lineWidth = Math.max(1, w*0.01);
  ctx.beginPath();
  ctx.moveTo(0, h*0.62); ctx.lineTo(w, h*0.62);
  ctx.stroke();
}

function addNoise(imgData, amount){
  const d = imgData.data;
  for(let i=0;i<d.length;i+=4){
    if(Math.random() < amount*0.06){
      const v = Math.random()<0.5 ? 0 : 255;
      d[i]=v; d[i+1]=v; d[i+2]=v;
    } else {
      const n = (Math.random()-0.5)*amount*70;
      d[i]=clamp(d[i]+n); d[i+1]=clamp(d[i+1]+n); d[i+2]=clamp(d[i+2]+n);
    }
  }
  return imgData;
}

/* ============================================================
   HERO CONVOLUTION ANIMATION
   ============================================================ */
(function heroAnim(){
  const canvas = document.getElementById('heroCanvas');
  const ctx = canvas.getContext('2d');
  const sumEl = document.getElementById('heroSum');
  const GRID = 8, CELL = 240/GRID;

  // synthetic 8x8 grayscale grid: dark left half, bright right, faint noise
  const values = [];
  for(let y=0;y<GRID;y++){
    const row=[];
    for(let x=0;x<GRID;x++){
      let base = x < GRID/2 ? 40 : 210;
      base += (Math.sin(x*1.3+y)*10);
      row.push(Math.round(Math.max(0,Math.min(255,base))));
    }
    values.push(row);
  }

  let pos = {r:0,c:0};
  let dir = 1;

  function draw(){
    ctx.clearRect(0,0,240,240);
    for(let y=0;y<GRID;y++){
      for(let x=0;x<GRID;x++){
        const v = values[y][x];
        ctx.fillStyle = `rgb(${v},${v},${v})`;
        ctx.fillRect(x*CELL, y*CELL, CELL, CELL);
      }
    }
    // grid lines
    ctx.strokeStyle = 'rgba(0,0,0,0.25)';
    for(let i=0;i<=GRID;i++){
      ctx.beginPath(); ctx.moveTo(i*CELL,0); ctx.lineTo(i*CELL,240); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0,i*CELL); ctx.lineTo(240,i*CELL); ctx.stroke();
    }
    // kernel overlay
    let sum = 0;
    const K = SOBEL_X;
    let ki=0;
    ctx.fillStyle = 'rgba(127,255,176,0.28)';
    ctx.fillRect(pos.c*CELL, pos.r*CELL, CELL*3, CELL*3);
    ctx.strokeStyle = '#7fffb0'; ctx.lineWidth=2;
    ctx.strokeRect(pos.c*CELL, pos.r*CELL, CELL*3, CELL*3);
    for(let dy=0; dy<3; dy++){
      for(let dx=0; dx<3; dx++){
        const yy = Math.min(GRID-1, pos.r+dy);
        const xx = Math.min(GRID-1, pos.c+dx);
        sum += values[yy][xx]*K[ki++];
      }
    }
    sumEl.textContent = 'Σ = ' + Math.round(sum);
  }

  function step(){
    pos.c += dir;
    if(pos.c > GRID-3 || pos.c < 0){
      dir *= -1;
      pos.c += dir;
      pos.r += 1;
      if(pos.r > GRID-3) pos.r = 0;
    }
    draw();
  }

  draw();
  setInterval(step, 750);
})();

/* ============================================================
   LAB CONTROLLER
   ============================================================ */
const TECHNIQUES = {
  mean:      { cat:'noise', label:'Mean', full:'Mean Filter',
               desc:'Averages every pixel with its neighbors — fast, simple, and effective on mild uniform noise, but it blurs edges without discrimination.',
               params:[{key:'radius', label:'Kernel radius', min:1,max:5,step:1,def:2}] },
  median:    { cat:'noise', label:'Median', full:'Median Filter',
               desc:'Takes the middle value of the sorted neighborhood instead of the average — excellent at removing salt-and-pepper noise while keeping edges sharp.',
               params:[{key:'radius', label:'Kernel radius', min:1,max:3,step:1,def:1}] },
  gaussian:  { cat:'noise', label:'Gaussian', full:'Gaussian Filter',
               desc:'Weights neighbors with a bell curve controlled by σ. The standard, well-behaved smoothing step used before gradient-based edge detection.',
               params:[{key:'sigma', label:'Sigma (σ)', min:0.5,max:4,step:0.1,def:1.4}] },
  sobel:     { cat:'gradient', label:'Sobel', full:'Sobel Operator',
               desc:'First-derivative directional operator with built-in smoothing (1-2-1 weighting). The most common general-purpose edge operator.',
               params:[] },
  prewitt:   { cat:'gradient', label:'Prewitt', full:'Prewitt Operator',
               desc:'First-derivative directional operator with uniform weighting — simpler than Sobel, and a little more sensitive to noise.',
               params:[] },
  laplacian: { cat:'gradient', label:'Laplacian', full:'Laplacian Operator',
               desc:'Second-derivative, direction-agnostic operator. Fires on edges of every orientation at once, at the cost of higher noise sensitivity.',
               params:[{key:'variant', label:'Neighborhood', type:'select', options:['4-neighbor','8-neighbor'], def:'4-neighbor'}] },
  canny:     { cat:'canny', label:'Canny', full:'Canny Edge Detector',
               desc:'Full pipeline: Gaussian smoothing → Sobel gradients → non-maximum suppression → double thresholding → hysteresis. Produces thin, connected edges.',
               params:[
                 {key:'sigma', label:'Gaussian σ', min:0.5,max:3,step:0.1,def:1.2},
                 {key:'low', label:'Low threshold', min:0,max:255,step:1,def:35},
                 {key:'high', label:'High threshold', min:0,max:255,step:1,def:90}
               ] }
};
const CATS = {
  noise: ['mean','median','gaussian'],
  gradient: ['sobel','prewitt','laplacian'],
  canny: ['canny']
};

const lab = {
  category: 'noise',
  technique: 'mean',
  params: {},
  sourceCanvas: document.getElementById('originalCanvas'),
  MAX_DIM: 320
};
Object.keys(TECHNIQUES).forEach(t => {
  lab.params[t] = {};
  TECHNIQUES[t].params.forEach(p => lab.params[t][p.key] = p.def);
});

function fitCanvasToSource(){
  // no-op helper placeholder (sizes are set at load time)
}

function loadImageToLab(img){
  let w = img.width, h = img.height;
  const scale = Math.min(1, lab.MAX_DIM / Math.max(w,h));
  w = Math.round(w*scale); h = Math.round(h*scale);
  const c = lab.sourceCanvas;
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);
  document.getElementById('inputPreview').width = w;
  document.getElementById('inputPreview').height = h;
  document.getElementById('inputPreview').getContext('2d').drawImage(c,0,0);
  runFilter();
}

function loadSample(){
  const w = 300, h = 220;
  const c = lab.sourceCanvas;
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  drawSample(ctx, w, h);
  document.getElementById('inputPreview').width = w;
  document.getElementById('inputPreview').height = h;
  document.getElementById('inputPreview').getContext('2d').drawImage(c,0,0);
  runFilter();
}

function currentImageData(){
  const c = lab.sourceCanvas;
  return c.getContext('2d').getImageData(0,0,c.width,c.height);
}

function setCanvasFromImageData(canvas, imgData){
  canvas.width = imgData.width; canvas.height = imgData.height;
  canvas.getContext('2d').putImageData(imgData, 0, 0);
}

function renderKernelMat(container, kernel, title){
  const block = document.createElement('div');
  block.className = 'kmat-block';
  if(title){
    const t = document.createElement('div');
    t.className = 'kmat-title'; t.textContent = title;
    block.appendChild(t);
  }
  const grid = document.createElement('div');
  grid.className = 'kmat';
  grid.style.gridTemplateColumns = 'repeat(3,1fr)';
  kernel.forEach(v => {
    const cell = document.createElement('div');
    cell.className = 'kcell';
    cell.textContent = v;
    grid.appendChild(cell);
  });
  block.appendChild(grid);
  container.appendChild(block);
}

function updateKernelDisplay(){
  const el = document.getElementById('kernelDisplay');
  el.innerHTML = '';
  const t = lab.technique;
  let label = null, pairDiv = null;
  if(t==='mean'){
    const r = lab.params.mean.radius;
    label = `Box kernel, ${r*2+1}×${r*2+1}, weight 1/${(r*2+1)*(r*2+1)}`;
  } else if(t==='gaussian'){
    label = `Separable Gaussian, σ=${lab.params.gaussian.sigma.toFixed(1)}`;
  } else if(t==='median'){
    const r = lab.params.median.radius;
    label = `Median over ${r*2+1}×${r*2+1} window (no fixed weights)`;
  } else if(t==='sobel' || t==='prewitt'){
    const l = document.createElement('div');
    l.className='label'; l.textContent = TECHNIQUES[t].full + ' kernels';
    el.appendChild(l);
    pairDiv = document.createElement('div'); pairDiv.className='kmat-pair';
    el.appendChild(pairDiv);
    const kx = t==='sobel'?SOBEL_X:PREWITT_X, ky = t==='sobel'?SOBEL_Y:PREWITT_Y;
    renderKernelMat(pairDiv, kx, 'Gx');
    renderKernelMat(pairDiv, ky, 'Gy');
    return;
  } else if(t==='laplacian'){
    const l = document.createElement('div');
    l.className='label'; l.textContent = 'Laplacian kernel';
    el.appendChild(l);
    pairDiv = document.createElement('div'); pairDiv.className='kmat-pair';
    el.appendChild(pairDiv);
    const variant = lab.params.laplacian.variant;
    renderKernelMat(pairDiv, variant==='4-neighbor'?LAP_4:LAP_8, variant);
    return;
  } else if(t==='canny'){
    label = 'Pipeline: Gaussian → Sobel → NMS → thresholds → hysteresis';
  }
  if(label){
    const l = document.createElement('div');
    l.className='label'; l.textContent = label;
    el.appendChild(l);
  }
}

function renderParams(){
  const container = document.getElementById('paramContainer');
  container.innerHTML = '';
  TECHNIQUES[lab.technique].params.forEach(p => {
    const row = document.createElement('div');
    row.className = 'param-row';
    if(p.type === 'select'){
      const label = document.createElement('label');
      label.textContent = p.label;
      row.appendChild(label);
      const sel = document.createElement('select');
      p.options.forEach(o => {
        const opt = document.createElement('option');
        opt.value = o; opt.textContent = o;
        if(o === lab.params[lab.technique][p.key]) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener('change', e => {
        lab.params[lab.technique][p.key] = e.target.value;
        updateKernelDisplay();
        runFilter();
      });
      row.appendChild(sel);
    } else {
      const label = document.createElement('label');
      const valSpan = document.createElement('span');
      valSpan.className='val';
      valSpan.textContent = lab.params[lab.technique][p.key];
      label.textContent = p.label + ' ';
      label.appendChild(valSpan);
      row.appendChild(label);
      const input = document.createElement('input');
      input.type = 'range'; input.min = p.min; input.max = p.max; input.step = p.step;
      input.value = lab.params[lab.technique][p.key];
      input.addEventListener('input', e => {
        const v = parseFloat(e.target.value);
        lab.params[lab.technique][p.key] = v;
        valSpan.textContent = Number.isInteger(p.step) ? v : v.toFixed(1);
        updateKernelDisplay();
        runFilter();
      });
      row.appendChild(input);
    }
    container.appendChild(row);
  });
}

function renderTechPills(){
  const wrap = document.getElementById('techPills');
  wrap.innerHTML = '';
  CATS[lab.category].forEach(t => {
    const btn = document.createElement('button');
    btn.className = 'tech-pill' + (t===lab.technique ? ' active' : '');
    btn.textContent = TECHNIQUES[t].label;
    btn.addEventListener('click', () => {
      lab.technique = t;
      renderTechPills();
      renderParams();
      updateKernelDisplay();
      document.getElementById('techDesc').innerHTML = '<b>' + TECHNIQUES[t].full + '.</b> ' + TECHNIQUES[t].desc;
      runFilter();
    });
    wrap.appendChild(btn);
  });
}

function runFilter(){
  const t = lab.technique;
  const src = currentImageData();
  const stageRow = document.getElementById('stageRow');
  stageRow.innerHTML = '';
  const outCanvas = document.getElementById('outputCanvas');

  if(t === 'mean'){
    setCanvasFromImageData(outCanvas, boxBlur(src, lab.params.mean.radius));
  } else if(t === 'median'){
    setCanvasFromImageData(outCanvas, medianFilter(src, lab.params.median.radius));
  } else if(t === 'gaussian'){
    setCanvasFromImageData(outCanvas, gaussianBlur(src, lab.params.gaussian.sigma));
  } else if(t === 'sobel' || t === 'prewitt'){
    const gray = toGray(src);
    const kx = t==='sobel'?SOBEL_X:PREWITT_X, ky = t==='sobel'?SOBEL_Y:PREWITT_Y;
    const {mag, max} = gradientMagnitude(gray, src.width, src.height, kx, ky);
    setCanvasFromImageData(outCanvas, normalizeToImageData(mag, src.width, src.height, max));
  } else if(t === 'laplacian'){
    const gray = toGray(src);
    const kernel = lab.params.laplacian.variant === '4-neighbor' ? LAP_4 : LAP_8;
    const res = convolveGray(gray, src.width, src.height, kernel);
    const absRes = res.map(Math.abs);
    let max=1; absRes.forEach(v=>{if(v>max)max=v;});
    setCanvasFromImageData(outCanvas, normalizeToImageData(absRes, src.width, src.height, max));
  } else if(t === 'canny'){
    const {sigma, low, high} = lab.params.canny;
    const res = canny(src, sigma, low, high);
    setCanvasFromImageData(outCanvas, res.edges);
    [['Blurred', res.blurred], ['Gradient magnitude', res.gradient], ['After NMS', res.nms]].forEach(([label, imgData]) => {
      const wrap = document.createElement('div');
      const lbl = document.createElement('div'); lbl.className='frame-label'; lbl.textContent = label;
      const frame = document.createElement('div'); frame.className='canvas-frame';
      const cv = document.createElement('canvas');
      frame.appendChild(cv);
      wrap.appendChild(lbl); wrap.appendChild(frame);
      stageRow.appendChild(wrap);
      setCanvasFromImageData(cv, imgData);
    });
  }
}

document.getElementById('catTabs').addEventListener('click', e => {
  const btn = e.target.closest('.cat-tab');
  if(!btn) return;
  document.querySelectorAll('.cat-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  lab.category = btn.dataset.cat;
  lab.technique = CATS[lab.category][0];
  renderTechPills();
  renderParams();
  updateKernelDisplay();
  document.getElementById('techDesc').innerHTML = '<b>' + TECHNIQUES[lab.technique].full + '.</b> ' + TECHNIQUES[lab.technique].desc;
  runFilter();
});

document.getElementById('sampleBtn').addEventListener('click', loadSample);
document.getElementById('noiseBtn').addEventListener('click', () => {
  const c = lab.sourceCanvas;
  const ctx = c.getContext('2d');
  const imgData = ctx.getImageData(0,0,c.width,c.height);
  addNoise(imgData, 1);
  ctx.putImageData(imgData, 0, 0);
  document.getElementById('inputPreview').getContext('2d').drawImage(c,0,0);
  runFilter();
});

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if(!file) return;
  const img = new Image();
  img.onload = () => loadImageToLab(img);
  img.src = URL.createObjectURL(file);
});
['dragover','dragenter'].forEach(evt => dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if(!file) return;
  const img = new Image();
  img.onload = () => loadImageToLab(img);
  img.src = URL.createObjectURL(file);
});

// init
renderTechPills();
renderParams();
updateKernelDisplay();
document.getElementById('techDesc').innerHTML = '<b>' + TECHNIQUES[lab.technique].full + '.</b> ' + TECHNIQUES[lab.technique].desc;
loadSample();