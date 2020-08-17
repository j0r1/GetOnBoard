const pieceCategories = [ 'pawn', 'rook', 'knight', 'bishop', 'queen', 'king' ];
const colorCategories = [ 'black', 'white' ];
const shortNames = {
    "black": {
        'pawn': 'p',
        'rook': 'r',
        'knight': 'n',
        'bishop': 'b',
        'queen': 'q',
        'king': 'k'
    },
    "white": {
        'pawn': 'P',
        'rook': 'R',
        'knight': 'N',
        'bishop': 'B',
        'queen': 'Q',
        'king': 'K'
    },
}

const categories = [];
for (const color of colorCategories)
    for (const piece of pieceCategories)
        categories.push(`${color} ${piece}`)
categories.push("empty")

function openTabForFen(fen)
{
    const url = "https://lichess.org/editor/" + fen;
    window.open(url);
}

function reduceSpaces(rank)
{
    let fen = "";
    let spaceCount = 0;
    for (let c of rank)
    {
        if (c != ' ')
        {
            if (spaceCount > 0)
            {
                fen += "" + spaceCount;
                spaceCount = 0;
            }
            fen += c;
        }
        else
            spaceCount++;
    }
    if (spaceCount > 0)
        fen += "" + spaceCount;
    return fen;
}

function analyzeOnePiece(model, piece)
{
    const divider = tf.tensor([255]).asType('float32');
    const pix = tf.browser.fromPixels(piece.getImageData(), 1).asType('float32').div(divider);
    const input = pix.as4D(1, ...pix.shape);
    const output = model.predict(input);
    const classes = output.as1D(output.shape[1]);
    const bestClass = tf.argMax(classes).bufferSync().get(0);
    // console.log(classes + " -> best: " + bestClass + " = " + categories[bestClass]);

    const className = categories[bestClass];
    if (className === "empty")
        return [ className, ' ' ];

    const [ color, name ] = className.split(" ");
    return [ className, shortNames[color][name] ];
}

function rescale(data, w, h, targetWH)
{
    const cnvs1 = new OffscreenCanvas(w, h);
    const ctx1 = cnvs1.getContext("2d");
    ctx1.putImageData(new ImageData(data, w, h), 0, 0);

    const cnvs2 = new OffscreenCanvas(targetWH, targetWH);
    const ctx2 = cnvs2.getContext("2d");
    ctx2.drawImage(cnvs1, 0, 0, w, h, 0, 0, targetWH, targetWH);
    
    const imgDat = ctx2.getImageData(0, 0, targetWH, targetWH);
    return new Pixels(targetWH, targetWH, imgDat.data);
}

async function processBoardParts(p, x0, x1, y0, y1, targetWH=32)
{
    const modelLocalStorage = 'indexeddb://piecedetecttionmodel';
    const modelUrl = "" + (new URL('jsmodel/model.json', import.meta.url));
    let model = null;
    let modelOrigin = null;

    for (const [ m, origin ]  of [ [ modelLocalStorage, 'localstorage'], 
                                   [ modelUrl, 'website' ] ])
    {
        try {
            model = await tf.loadLayersModel(m);
            modelOrigin = origin;
            break;
        } catch (e) {
            console.warn("Temporary error: " + e);
        }
    }

    if (!model)
        throw "No model could be loaded";
    
    console.log(`Model loaded from ${modelOrigin}!`);
    if (modelOrigin !== 'localstorage')
    {
        console.log("Saving model to local storage");
        await model.save(modelLocalStorage);
    }

    const dx = (x1-x0)/8;
    const dy = (y1-y0)/8;
    // const board = new Pixels(... p.getRGBA(x0, y0, x1-x0, y1-y0));
    // board.showInNewTab("Board part");

    const fetchBoardPiece = (X, Y) => {

        const left = Math.floor(x0 + X*dx);
        const right = Math.floor(x0 + (X+1)*dx);
        const top = Math.floor(y0 + Y*dy);
        const bottom = Math.floor(y0 + (Y+1)*dy);
        const w = right-left;
        const h = bottom-top;

        return rescale(p.getRGBAGray(left, top, w, h)[2], w, h, targetWH);
    }

    let pieceImages = [];
    const colName = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    let fen = "";
    for (let Y = 0 ; Y < 8 ; Y++)
    {
        let rank = "";
        for (let X = 0 ; X < 8 ; X++)
        {
            const piece = fetchBoardPiece(X, Y);
            const [ fullName, shortName ] = analyzeOnePiece(model, piece);
            pieceImages.push({
                "position": colName[X] + (8-Y),
                "piece": piece,
                "fullname": fullName,
                "shortname": shortName,
            })
            rank += shortName;
        }
        console.log(rank);
        if (fen.length != 0)
            fen += '/';
        fen += reduceSpaces(rank);
    }
    openTabForFen(fen);
}

class Pixels
{
    constructor(w, h, pixels, flip = false)
    {
        this.width = w;
        this.height = h;
        this.pixels = pixels;
        this.flip = flip;
    }

    getWidth()
    {
        return this.width;
    }

    getHeight()
    {
        return this.height;
    }

    getPixel(x, y)
    {
        const [ r, g, b, a ] = this.getPixelRGBA(x, y);
        return r*256*256*256 + g*256*256 + b*256 + a;
    }

    getPixelRGBA(x, y)
    {
        if (this.flip)
            y = this.height-1-y;
        const offset = (this.width*4)*y + (x*4);
        return [ this.pixels[offset+0], this.pixels[offset+1], this.pixels[offset+2], this.pixels[offset+3] ];
    }
    
    getPixelGray(x, y)
    {
        const [ r, g, b, a ] = this.getPixelRGBA(x, y);
        const gray = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
        if (gray < 0)
            gray = 0;
        else if (gray > 255)
            gray = 255;
        return gray;
    }

    getImageData()
    {
        return new ImageData(this.getRGBA()[2], this.getWidth(), this.getHeight());
    }

    getRGBA(x0 = 0, y0 = 0, w = -1, h = -1)
    {
        return this._getRGBAHelper(x0, y0, w, h, (x, y) => { return this.getPixelRGBA(x, y) });
    }

    getRGBAGray(x0 = 0, y0 = 0, w = -1, h = -1)
    {
        return this._getRGBAHelper(x0, y0, w, h, (x, y) => { 
            const g = this.getPixelGray(x, y);
            return [ g, g, g, 255 ];
        });
    }

    _getRGBAHelper(x0, y0, w, h, fn)
    {
        if (w < 0)
            w = this.width;
        if (h < 0)
            h = this.height;

        const data = new Uint8ClampedArray(w*h*4);

        for (let y = 0 ; y < h ; y++)
        {
            for (let x = 0 ; x < w ; x++)
            {
                const [ r, g, b, a ] = fn(x+x0, y+y0);
                const offset = (x*4) + (w*4*y);
                data[offset+0] = r;
                data[offset+1] = g;
                data[offset+2] = b;
                data[offset+3] = a;
            }
        }
        return [w, h, data];
    }

    getDataURL(x0 = 0, y0 = 0, w = -1, h = -1)
    {
        let data = null;
        [w, h, data ] = this.getRGBA(x0, y0, w, h);

        const cnvs = document.createElement("canvas");
        cnvs.width = w;
        cnvs.height = h;
        const ctx = cnvs.getContext("2d");
        const imgData = new ImageData(data, w, h);
        ctx.putImageData(imgData, 0, 0);
        return cnvs.toDataURL();
    }

    showInNewTab(title = "")
    {
        const url = this.getDataURL();
        const newTab = window.open();
        newTab.document.body.innerHTML = `<H1>${title}</H1><img src="${url}" width="${this.getWidth()}" height="${this.getHeight()}">`;
        return newTab;
    }
}

function analyzePixels(p)
{
    const bgCol = p.getPixel(0, 0);
    const halfY = Math.floor(p.getHeight()/2);
    
    let leftX = -1;
    for (let x = 0 ; x < p.getWidth() ; x++)
    {
        if (p.getPixel(x, halfY) !== bgCol)
        {
            leftX = x;
            break;
        }
    }

    if (leftX < 0)
        throw "No left border found";
    
    console.log("LeftX: " + leftX);

    let rightX = -1;
    for (let x = leftX ; x < p.getWidth() ; x++)
    {
        if (p.getPixel(x, halfY) === bgCol)
        {
            rightX = x;
            break;
        }
    }

    if (rightX < 0)
        throw "No right border found";

    console.log("RightX: " + rightX);

    let partX = Math.floor(leftX + (rightX-leftX)/4);
    let topY = -1;
    for (let y = 0 ; y < p.getHeight() ; y++)
    {
        if (p.getPixel(partX, y) !== bgCol)
        {
            topY = y;
            break;
        }
    }

    if (topY < 0)
        throw "No top border found";

    console.log("TopY: " + topY);

    let bottomY = -1;
    for (let y = topY ; y < p.getHeight() ; y++)
    {
        if (p.getPixel(partX, y) === bgCol)
        {
            bottomY = y;
            break;
        }
    }

    if (bottomY < 0)
        throw "No bottom border found";

    console.log("BottomY: " + bottomY);

    const width = (rightX-leftX);
    const height = (bottomY-topY);
    const avgSize = (width+height)/2;
    const borderPix = Math.round(avgSize/50);
    console.log("borderPix = " + borderPix);
    
    processBoardParts(p, leftX+borderPix, rightX-borderPix, topY+borderPix, bottomY-borderPix).then(() => {
        console.log("Board parts processed");
    }).catch((err) => {
        console.error("Error:" + err);
    })
}

function readData()
{
    const cnvs = document.getElementsByTagName("canvas")[0];
    const gl = cnvs.getContext("webgl");

    const w = cnvs.width;
    const h = cnvs.height;    
    const pixels = new Uint8Array(w*h*4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    // console.log(pixels);
    return [w, h, pixels];
}

function main()
{
    if (!document.getElementById("tsscript"))
    {
        let script = document.createElement("script");
        script.setAttribute("id", "tsscript");
        script.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js";
        document.body.appendChild(script);
    }

    console.log("main(module)");
    const cnvs = document.getElementsByTagName("canvas")[0];
    const gl = cnvs.getContext("webgl");

    if (!gl._realDrawElements)
        gl._realDrawElements = gl.drawElements;

    const neededDrawElementsCount = 3; // Entire update seems to take three updates
    gl._drawElementsCount = 0;
    gl.drawElements = function(mode, count, type, offset)
    {
        console.log("drawElements");
        gl._realDrawElements(mode, count, type, offset);
        gl._drawElementsCount += 1;

        if (gl._drawElementsCount < neededDrawElementsCount)
            return;

        const [w, h, pixels] = readData();
        // console.log("w = " + w + " h = " + h + " pixels:");
        // console.log(pixels);
        
        setTimeout(() => { 
            const p = new Pixels(w, h, pixels, true);
            // p.showInNewTab("Entire board");
            analyzePixels(p); 
        }, 0);

        gl.drawElements = gl._realDrawElements;
        delete gl._realDrawElements;
    }
}

console.log("Main module");
main();
