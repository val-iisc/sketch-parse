var _DEBUG = {};

$(document).ready(function() {

    /* variables */
    var gPaths;
    var canvas;
    var gTool, zTool;

    var cursors = {
    near_endpt: "url(http://www.rw-designer.com/cursor-extern.php?id=84706), auto",
    draw: "url(http://www.rw-designer.com/cursor-extern.php?id=84708), auto"
    };

    /* setup canvas */
    canvas = document.getElementById("myCanvas");
    canvas.style.cursor = cursors.draw;

    paper.setup(canvas);
    zTool = new Tool(); zTool.activate();
    gTool = new Tool();

    /* import svg */
    project.importSVG($("svg").get(0), {
        expandShapes: true,
        onLoad: function(group) {

            group.visible = true;
            gPaths = group.getItems({   // get paths
                class: Path
            });
            gPaths.map(function(path) {
                path.strokeWidth = 4;
                path.selected = false;
            });
        }
    });

    var sketchr = new Sketch(gTool, zTool);

    /* Setup draw Tool */
    const magThreshold     = 15;
    const edgeThreshold    = 10;

    gTool.minDistance       = 10;
    gTool.distanceThreshold = 5;

    gTool.onMouseDown = function(event) {
        if(Sketch.activePath === null) {
            Sketch.activePath = new Path({
                strokeWidth: 4,
                selected: true,
                strokeColor: Sketch.activePart.color
            }); 

            Sketch.activePath.add(event.point);
        }
    };

    gTool.onMouseDrag = function(event) {
        var nearestPt, minDist = Number.POSITIVE_INFINITY;
        for(var idx = 0; idx < gPaths.length; ++idx) {
            let pt = gPaths[idx].getNearestPoint(event.point);
            let vec = pt.subtract(event.point);
            if(minDist > vec.length) {
                minDist = vec.length;
                nearestPt = pt;
            }
        }
        if(minDist < edgeThreshold) {
            Sketch.activePath.add(nearestPt);
        } else {
            Sketch.activePath.add(event.point);
        }
    };

    gTool.onMouseUp = function(event) {
        var vec = event.point.subtract(Sketch.activePath.firstSegment._point);
        if(vec.length > magThreshold) {
            Sketch.activePath.add(event.point);
            Sketch.activePath.reduce();
        } else {
            Sketch.activePath.strokeColor = Sketch.activePart.color;
            Sketch.activePath.fillColor = Sketch.activePart.color
            Sketch.activePath.fillColor.alpha = 0.3;
            sketchr.pprocess();
            sketchr.disable();
            sketchr.incstatus(Sketch.activePart.part_name);
        }
    };
    _DEBUG.sketchr = sketchr;
    _DEBUG.activePath = Sketch.activePath;
    _DEBUG.activePart = Sketch.activePart;



});

