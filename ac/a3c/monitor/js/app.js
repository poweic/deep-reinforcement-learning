var fields = ["returns", "advantages", "vehicle_state", "prev_reward", "front_view", "actions", "prev_action"];

var data = {};

const K = 200;
const MULTIPLIER = 20;
const dt = 0.01;
const wheelbase = 2.0;
const R = 40;
const vehicle_size = 10;

function to_image_coord(x, y) {
  return {'x': (x + 10) * K, 'y': (20 - y) * K};
}

function render_front_view(img) {
  var s = Snap('#frontview');
  const W = 20;
  for (var i=0; i<img.length; ++i) {
    for (var j=0; j<img[i].length; ++j) {
      var v = img[j][i][0];
      s.rect(i * W, j * W, W, W).attr({
	fill: sprintf('rgb(%d, %d, %d)', v, v, v)
      });
    }
  }
}

function draw_all() {

  var n_agents = data['returns'].length;
  var T = data['returns'][0].length;

  var s = Snap(".canvas");

  // For each agent
  data['vehicle_state'].forEach(function (states, j) {

    var actions = data['actions'][j];
    var front_views = data['front_view'][j];

    var g = s.group().addClass("agent agent-" + j);

    var path = "";
    var points = [];
    var pt_vxs = [];
    var pt_vys = [];
    var pt_ays = [];

    // For each timestep
    states.forEach(function(state, i) {

      // Get x, y, theta, x', y', theta'
      var x     = state[0];
      var y     = state[1];
      var theta = state[2];
      var vx    = state[3];
      var vy    = state[4];
      var omega = state[5];

      // Get actions
      var action = actions[i];
      var ay      = action[0];
      var a_steer = action[1];

      // Compute cos(theta), sin(theta), and the dx, dy
      var cos = Math.cos(theta);
      var sin = Math.sin(theta);

      dx = (cos * vx - sin * vy) * dt;
      dy = (sin * vx + cos * vy) * dt;

      var pt = to_image_coord(x, y);
      var pt_plus = to_image_coord(x+dx, y+dy);
      var pt_minus = to_image_coord(x-dx, y-dy);

      dx_vx = (cos * vx) * dt;
      dy_vx = (sin * vx) * dt;
      dx_vy = (-sin * vy) * dt;
      dy_vy = (+cos * vy) * dt;
      dx_ay = -sin * ay * dt;
      dy_ay = +cos * ay * dt;

      var pt_vx = to_image_coord(x+dx_vx*MULTIPLIER, y+dy_vx*MULTIPLIER);
      var pt_vy = to_image_coord(x+dx_vy*MULTIPLIER, y+dy_vy*MULTIPLIER);
      var pt_ay = to_image_coord(x+dx_ay*MULTIPLIER, y+dy_ay*MULTIPLIER);
      // console.log(pt, pt_plus, pt_minus, cos, sin);

      points.push(pt);
      pt_vxs.push(pt_vx);
      pt_vys.push(pt_vy);
      pt_ays.push(pt_ay);

      if (i == 0) {
	path += sprintf("M%.2f %.2f C %.2f %.2f ", pt_plus.x, pt_plus.y, pt.x, pt.y);
      }
      else {
	cmd = (i == 1) ? "," : "S";
	path += sprintf("%s %.2f %.2f, %.2f %.2f ", cmd, pt_minus.x, pt_minus.y, pt.x, pt.y);
      }
    });

    g.path(path).addClass("trajectory");

    function arc_path(pt, r, theta1, theta2) {
      var dtheta = theta2 - theta1;
      return sprintf("M%.2f %.2f A %d %d, 0, %s, %.2f %.2f",
	pt.x - r * Math.sin(theta1),  pt.y - r * Math.cos(theta1),
	r, r,
	(dtheta < 0) ? "0 1" : ((dtheta < Math.PI) ? "0 0" : "1 1"),
	pt.x - r * Math.sin(theta2),  pt.y - r * Math.cos(theta2)
      );
    }

    function mod_2pi(theta) { return theta % (2 * Math.PI); }

    var last = pt_vys[pt_vys.length - 1];
    g.text(last.x, last.y, sprintf("Agent %02d", j)).addClass("agent-name");

    // For each timestep again
    points.forEach(function (pt, i) {
      // Draw vehicle position
      g.circle(pt.x, pt.y, vehicle_size).addClass("pose-xy").attr({
	'data-i': i,
	'data-j': j
      }).mouseover(function (event) {
	var i = parseInt(event.target.getAttribute('data-i'));
	var j = parseInt(event.target.getAttribute('data-j'));
	render_front_view(data['front_view'][j][i])
      });

      g.text(pt.x + 10, pt.y - 10, sprintf("t=%02d", i)).addClass("agent-name");

      // Draw vx, vy and omega (theta is included in vy's direction)
      var theta = states[i][2];
      var omega = states[i][5];
      g.line(pt.x, pt.y, pt_vxs[i].x, pt_vxs[i].y).addClass("pose pose-vx");
      g.line(pt.x, pt.y, pt_vys[i].x, pt_vys[i].y).addClass("pose pose-vy");
      g.path(arc_path(pt, R, theta, theta + omega)).addClass("pose pose-yawrate");

      // Draw actions (Negate because YVCA has weird convention: right > 0, left < 0)
      var a_yawrate = actions[i][1] * -1;

      // Turn yawrate to steering angle
      /* var vf = states[i][4];
      var a_steer = Math.atan(wheelbase * a_yawrate / vf);
      a_steer = mod_2pi(a_steer); */

      g.line(pt_vys[i].x, pt_vys[i].y, pt_ays[i].x, pt_ays[i].y).addClass("action action-vy");
      g.path(arc_path(pt, R*1.2, theta, theta + a_yawrate)).addClass("action action-steer");
    });

  });

  /* new Vivus('canvas', {
    duration: 200,
    animTimingFunction: Vivus.EASE
  }); */
}

function request_data(callback) {
  var semaphore = 0;

  for (var i=0; i<fields.length; ++i) {
    semaphore += 1;
    $.get("/data/" + fields[i], (function(field) {
      return function (res) {
	data[field] = JSON.parse(res);
	semaphore -= 1;

	if (semaphore == 0)
	  callback();
      };
    })(fields[i]));
  }

}

function main() {
  draw_all();

  var minimap = $('.container').clone();
  $('body').append(minimap);
  minimap.addClass('minimap');
  Snap('.container.minimap .canvas').circle(0, 0, 50).addClass('mouse');

  $('.container:not(.minimap) .canvas').mousemove(function (event) {
    var x = event.offsetX;
    var y = event.offsetY;
    var s = Snap('.container.minimap .canvas');
    var circle = s.select('circle.mouse').attr({
      cx: x, cy: y
    });
    console.log(x, y);
  });
}

request_data(main);
