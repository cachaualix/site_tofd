(() => {
  const W = 360, H = 600, T1 = 380, T2 = 490, XC = 180;
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
  const wav = t => Math.cos(2 * Math.PI * t / 14) * Math.exp(-((t / 9) ** 2));

  // Données simulées : colonne = A-scan, ligne = échantillon de profondeur
  const data = Array.from({ length: W }, (_, x) => {
    const a = new Float32Array(H);
    for (let j = 0; j < H; j++) a[j] = 0.12 * gauss();
    const amp = Math.exp(-(((x - XC) / 70) ** 2));
    const add = (t0, A) => {
      const ti = Math.sqrt(t0 * t0 + (0.9 * (x - XC)) ** 2);
      for (let j = Math.max(0, Math.floor(ti - 30)); j < Math.min(H, ti + 30); j++) a[j] += A * amp * wav(j - ti);
    };
    add(T1, 0.9); add(T2, -0.8);
    for (let j = 0; j < H; j++) {
      a[j] += 1.0 * wav(j - 140) + 1.1 * wav(j - 560);
    }
    return a;
  });

  const envelope = a => {
    const e = new Float32Array(H), k = 4;
    for (let j = 0; j < H; j++) {
      let s = 0, n = 0;
      for (let i = Math.max(0, j - k); i <= Math.min(H - 1, j + k); i++) { s += Math.abs(a[i]); n++; }
      e[j] = s / n;
    }
    return e;
  };

  // Détection : plus grand pic, puis second pic à au moins 25 échantillons, entre onde latérale et fond
  const detect = x => {
    const e = envelope(data[x]), lo = 175, hi = 530, thr = 0.28;
    let p1 = -1;
    for (let j = lo; j < hi; j++) if (e[j] > thr && (p1 < 0 || e[j] > e[p1])) p1 = j;
    if (p1 < 0) return { env: e };
    let p2 = -1;
    for (let j = lo; j < hi; j++) if (Math.abs(j - p1) >= 25 && e[j] > thr && (p2 < 0 || e[j] > e[p2])) p2 = j;
    if (p2 < 0) return { env: e };
    return { env: e, top: Math.min(p1, p2), bot: Math.max(p1, p2) };
  };

  const bs = document.getElementById("bscan"), as = document.getElementById("ascan");
  const bc = bs.getContext("2d"), ac = as.getContext("2d");
  const off = document.createElement("canvas"); off.width = W; off.height = H;
  const img = off.getContext("2d").createImageData(W, H);
  for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
    const v = 128 + 110 * Math.tanh(data[x][y] * 1.4), i = 4 * (y * W + x);
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255;
  }
  off.getContext("2d").putImageData(img, 0, 0);

  const pos = document.getElementById("pos"), out = document.getElementById("posOut");
  const rT = document.getElementById("rTop"), rB = document.getElementById("rBot"), rG = document.getElementById("rGap");
  const GREEN = "#3ddc84", MAG = "#ff4fd8";

  const draw = () => {
    const x = +pos.value, d = detect(x), sx = bs.width / W, sy = bs.height / H;
    out.textContent = x;
    bc.drawImage(off, 0, 0, bs.width, bs.height);
    bc.strokeStyle = "#ffb454"; bc.setLineDash([3, 4]); bc.lineWidth = 1;
    bc.beginPath(); bc.moveTo(x * sx, 0); bc.lineTo(x * sx, bs.height); bc.stroke();
    bc.setLineDash([6, 4]); bc.lineWidth = 1.6;
    if (d.top !== undefined) {
      bc.strokeStyle = GREEN; bc.beginPath(); bc.moveTo(0, d.top * sy); bc.lineTo(bs.width, d.top * sy); bc.stroke();
      bc.strokeStyle = MAG; bc.beginPath(); bc.moveTo(0, d.bot * sy); bc.lineTo(bs.width, d.bot * sy); bc.stroke();
    }
    bc.setLineDash([]); bc.fillStyle = "#fff"; bc.font = "12px 'Public Sans',sans-serif";
    bc.fillText("Position de la sonde →", 8, bs.height - 8);
    bc.save(); bc.translate(14, 130); bc.rotate(-Math.PI / 2); bc.restore();

    // A-scan : profondeur en vertical, amplitude en horizontal
    ac.fillStyle = "#111"; ac.fillRect(0, 0, as.width, as.height);
    const cx = as.width / 2, k = as.height / H;
    ac.strokeStyle = "#666"; ac.lineWidth = 1; ac.beginPath(); ac.moveTo(cx, 0); ac.lineTo(cx, as.height); ac.stroke();
    ac.strokeStyle = "#d8e0e6"; ac.beginPath();
    for (let j = 0; j < H; j++) { const px = cx + data[x][j] * 40; j ? ac.lineTo(px, j * k) : ac.moveTo(px, 0); }
    ac.stroke();
    ac.strokeStyle = "#ffb454"; ac.beginPath();
    for (let j = 0; j < H; j++) { const px = cx + d.env[j] * 80; j ? ac.lineTo(px, j * k) : ac.moveTo(px, 0); }
    ac.stroke();
    if (d.top !== undefined) {
      [[d.top, GREEN], [d.bot, MAG]].forEach(([t, c]) => {
        ac.strokeStyle = c; ac.setLineDash([6, 4]); ac.lineWidth = 1.6;
        ac.beginPath(); ac.moveTo(0, t * k); ac.lineTo(as.width, t * k); ac.stroke();
      });
      ac.setLineDash([]);
    }
    ac.fillStyle = "#fff"; ac.font = "12px 'Public Sans',sans-serif";
    ac.fillText("A-scan et enveloppe", 8, 16);

    if (d.top === undefined) { rT.textContent = rB.textContent = rG.textContent = "non détecté"; }
    else { rT.textContent = d.top; rB.textContent = d.bot; rG.textContent = d.bot - d.top + " éch."; }
  };

  pos.addEventListener("input", draw);
  draw();
  document.getElementById("year").textContent = new Date().getFullYear();
})();
