SERVER = 'http://localhost:5000'
API_ENDPOINT = SERVER + '/api/v1.0'

$console = $('#console');

var task_walk = function(task, depth) {
    for (var i = 0; i < depth; i++) {
        $console.append('  ');
    }
    $console.append(task.body.data.split('\n')[0] + '\n');

    for (var i = 0; i < task.children.length; i++) {
        task_walk(task.children[i], depth + 1);
    }
}

$.get(
    API_ENDPOINT + '/tasks',
    {
    },
    function(data) {
        for (var i = 0; i < data.length; i++) {
            task_walk(data[i], 0);
        }
    }
);
