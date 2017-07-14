/* jshint esversion: 6 */

class Sketch {
    constructor(drawTool, panTool) {
        Sketch.activePath = null;
        Sketch.activePart = null;
        Sketch.drawTool = drawTool;
        Sketch.panTool = panTool;
        var self = this;

        $( "div.part-add" ).on("click", function(eventObject) {
            var element = $( this );
            Sketch.activePart = {
                parent : element.parent("div.part-container").eq(0),
                part_name : element.siblings(".part-name").eq(0).text().trim(),
                status : element.siblings(".part-status").eq(0),
                color : element.siblings(".part-color").eq(0).attr('id'),
            };

            self.setup();
        });


        $( "div.part-remove" ).on("click", function(eventObject) {
           var element = $( this ); 
           var part_name = self.get_part_name(element);
           if(Sketch.partPaths[part_name].length > 0) {
               var part_path = Sketch.partPaths[part_name].pop();
               part_path.remove();
               self.decstatus(part_name);
            }
        });

        $( "#delete-button").on("click", function(eventObject) {
            if (Sketch.activePath !== null) {
                Sketch.activePath.remove();
                Sketch.activePath = null;
                disable();
            }
        });

        $( "#complete-button").on("click", function(eventObject) {
            if (Sketch.activePath !== null) {
                self.pprocess();
                self.incstatus(Sketch.activePart.part_name);
                self.disable();
            }
        });

        $( "#undo-button").on("click", function(eventObject) {
            if (Sketch.activePath !== null) {
                Sketch.activePath.lastSegment.remove();
            }
            if (Sketch.activePath.segments.length === 0) {
                Sketch.activePath.remove();
                self.disable();
            }
        });

        /* setup the part paths object */
        Sketch.partPaths = {};
        $( "div.part-name" ).each(function() {
            Sketch.partPaths[$(this).text().trim()] = [];
        });
    } 

    setup() {
        $( "#draw-button").removeAttr("disabled");
        $( "#complete-button").removeAttr("disabled");
        $( "#delete-button").removeAttr("disabled");
        $( "#undo-button").removeAttr("disabled");
        Sketch.drawTool.activate();
    }

    pprocess() {
        Sketch.activePath.closePath();
        Sketch.activePath.reduce();
        Sketch.activePath.selected = false;
        Sketch.partPaths[Sketch.activePart.part_name].push(Sketch.activePath);
    }

    incstatus(part_name) {
        var id = '#' + part_name + '-status';
        var part_status = $(id).eq(0);
        part_status.text(Number(part_status.text().trim()) + 1);
    }

    decstatus(part_name) {
        var id = '#' + part_name + '-status';
        var part_status = $(id).eq(0);
        part_status.text(Number(part_status.text().trim()) - 1);
    }

    get_part_name(element) {
        return element.siblings(".part-name").eq(0).text().trim();
    }

    disable() {
        $( "#draw-button").attr("disabled", "disabled");
        $( "#complete-button").attr("disabled", "disabled");
        $( "#delete-button").attr("disabled", "disabled");
        $( "#undo-button").attr("disabled", "disabled");
        Sketch.activePath = null;
        Sketch.panTool.activate();
    }


}

/* |draw|       (d1)
 * |delete|     (d2)
 * |undo tool|   (d3)
 * |complete|   (d4)
 *|s1| |part-name| (p1) |+| (b1) |-| (b2)
 * ...
 * ...
 * ...
 *
 *
 *                                  |submit| (Sf)
 *
 * ## Environment ##
 * activePath
 * partArray
 * */

/*
 * b1.click -> d1.activate(), pcur = p1, activePath = new()
 * d2.click -> d1.deactivate(), pcur.deactivate(), activePath.remove()
 * d3.click -> activePath--; if (activePath.length == 1) d1.deactivate(), pcur.deactivate(), activePath.remove()
 * d4.click -> pcur.s++, pcur.deactivate(), d1.deactivate(), partArray[pcur.idx].push(activePath), activePath.deactivate()
 * activePath.close? -> d4.click
 * b2.click -> partArray[b2.part].pop(), b2.s--
 */
