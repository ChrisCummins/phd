SERVER = 'http://localhost:5000'
API_ENDPOINT = SERVER + '/api/v1.0'

$tasks = $('#app .tasks');
$details = $('#app .details');

var generate_task_list = function(tasks) {

  var walker = function(task, html) {
    html += '<li>';

    if (task.children.length) {
      html += '<i class="fa fa-caret-down see-more down" aria-hidden="true"></i>';
    }

    html += '<img src="/img/task.svg" /><span>' + task.body.data.split('\n')[0] +
      '</span><hr/>';

    if (task.children.length) {
      html += '<ul>';
      for (var i = 0; i < task.children.length; i++) {
        html = walker(task.children[i], html);
      }
      html += '</ul>';
    }

    return html + '</li>';
  }

  var ret = '<ul>';
  for (var i = 0; i < tasks.length; i++) {
    ret = walker(tasks[i], ret);
  }

  return ret + '</ul>';
}

console.log("GET /tasks");
$.get(
  API_ENDPOINT + '/tasks', {},
  function(data) {
    console.log('/tasks');
    $tasks.html(generate_task_list(data));
  }
);

console.log("GET /tasks/17");
$.get(
  API_ENDPOINT + '/tasks/17', {},
  function(data) {
    console.log('/tasks/17');

    $('.details .task-header').html(markdown.toHTML(data.body.split('\n')[0]))
    $('.details .task-body').html(markdown.toHTML(data.body))
  }
);