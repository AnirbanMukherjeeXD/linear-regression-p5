
let x = [];
let y = [];

let m,b;
let ll;
mse=0;

function clearAll(){
  x=[];
  y=[];
  mse=0;

}

function lrSet(x,n){
  if(n==0){
    document.getElementById('lr').value=x;  
  }else{
    document.getElementById('lr_slide').value=x*100;  
  }
  
}

function setup() {
  //var cnv = createCanvas(windowWidth,360);
  var cnv = createCanvas(900,360);
  cnv.style('display', 'block');
  cnv.parent('graph');
  background(50);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));  
}

function loss(pred,labels){
  
  ll=pred.sub(labels).square().mean();
  //ll.print();
  mse=Number(ll.dataSync()).toFixed(5);
  if(mse==0)
    mse=Number(mse).toFixed(0);
  //console.log(typeof(Number(mse)));
  return ll;
}

function predict(x){
  const tx=tf.tensor1d(x);
  y_pred=tx.mul(m).add(b);

  return y_pred;
}

function mousePressed() {
  if(mouseX>0&&mouseX<900&&mouseY>0&&mouseY<360) {
    //Var needed ?
    var a = map(mouseX, 0, width, 0, 1);
    var b = map(mouseY, 0, height, 1, 0);

    x.push(a);
    y.push(b);
    //console.log(a,",",b);
  }

}

function draw(){
  //console.log(m,b)
  const learningRate = document.getElementById('lr').value;
  optimizer=tf.train.sgd(learningRate);
  tf.tidy(() => {
    if(x.length>0){
      const ty=tf.tensor1d(y);
      optimizer.minimize(() => loss(predict(x), ty));
    }
  });
  background(50);
  stroke(20);
  strokeWeight(0);
  textSize(30);
  textFont('Tw Cen MT');
  fill(200);
  txt="MSE="+mse;
  text(txt,50,80);
  
  stroke(255);
  strokeWeight(8);
  for(i=0;i<x.length;i++){
    let px=map(x[i],0,1,0,width);
    let py=map(y[i],0,1,height,0);
    point(px,py);
  }

  const lineX=[0, 1];
  const ys = tf.tidy(() => predict(lineX));
  let lineY=ys.dataSync();
  ys.dispose();

  let x1=map(lineX[0],0,1,0,width);
  let x2=map(lineX[1],0,1,0,width);

  let y1=map(lineY[0],0,1,height,0);
  let y2=map(lineY[1],0,1,height,0);
  strokeWeight(2);
  if(x.length>0){
    line(x1,y1,x2,y2);
  }
  //document.getElementById('mse').value=mouseY;

}