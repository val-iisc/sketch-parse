$(document).ready(function() {
    $("#form-done").submit(function() {
        var svg_partPaths = JSON.parse(JSON.stringify(Sketch.partPaths));
        for(var key in Sketch.partPaths) {
            for (var i = 0; i < Sketch.partPaths[key].length; ++i) {
                var svg_path = Sketch.partPaths[key][i].exportSVG();
                svg_partPaths[key][i] = svg_path.outerHTML;
            }
        }

    var json_paths = $('<input>', {
                    type: 'hidden',
                    id: 'partPaths',
                    name: 'part_paths',
                    value: JSON.stringify(Sketch.partPaths)});

    var svg_paths = $('<input>', {
                    type: 'hidden',
                    id: 'svg_partPaths',
                    name: 'svg_part_paths',
                    value: JSON.stringify(svg_partPaths)});
        $("#form-done").append(json_paths);
        $("#form-done").append(svg_paths);
    });
});
