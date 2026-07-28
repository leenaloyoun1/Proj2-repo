from flask import (
    Flask,
    render_template,
    request
)

from graph import run_task_planner


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":

        user_tasks = request.form.get(
            "tasks"
        )

        if user_tasks:

            result = run_task_planner(
                user_tasks
            )
            result["original_input"] = user_tasks

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )
