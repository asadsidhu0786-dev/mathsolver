from flask import Flask, request, render_template_string, send_file
import io
import matplotlib.pyplot as plt
import sympy as sp

app = Flask(__name__)

# HTML template (simple)
HTML = """
<!doctype html>
<title>Math Solver</title>
<h1>Math Solver</h1>
<form method="post" action="/">
  <label for="expr">Expression / Problem:</label><br>
  <input type="text" id="expr" name="expr" size="80" value="{{expr|default('')}}"><br><br>

  <label for="op">Operation:</label>
  <select id="op" name="op">
    <option value="simplify" {{'selected' if op=='simplify' else ''}}>Simplify</option>
    <option value="evaluate" {{'selected' if op=='evaluate' else ''}}>Evaluate (numeric)</option>
    <option value="differentiate" {{'selected' if op=='differentiate' else ''}}>Differentiate</option>
    <option value="integrate" {{'selected' if op=='integrate' else ''}}>Integrate</option>
    <option value="solve" {{'selected' if op=='solve' else ''}}>Solve equation</option>
    <option value="plot" {{'selected' if op=='plot' else ''}}>Plot (y=...)</option>
  </select>

  <label for="var">Variable (for diff/integrate/solve/plot):</label>
  <input type="text" id="var" name="var" size="5" value="{{var|default('x')}}">

  <label for="set_prec">Numeric precision (for eval):</label>
  <input type="number" id="set_prec" name="set_prec" min="1" max="50" value="{{set_prec|default(15)}}">

  <button type="submit">Solve</button>
</form>

{% if error %}
  <h3 style="color:darkred">Error: {{error}}</h3>
{% endif %}

{% if result %}
  <h3>Result</h3>
  <pre>{{result}}</pre>
{% endif %}

{% if plot_available %}
  <h3>Plot</h3>
  <img src="/plot.png?expr={{expr|urlencode}}&var={{var|urlencode}}">
{% endif %}

<hr>
<p>Examples:
<ul>
  <li><code>2+3*4</code></li>
  <li><code>sin(x)**2 + cos(x)**2</code></li>
  <li><code>diff: x**2 + 3*x + 1</code> (use operation "Differentiate")</li>
  <li><code>integrate: exp(x)</code> (use "Integrate")</li>
  <li><code>x**2 - 5*x + 6 = 0</code> (use "Solve equation")</li>
  <li><code>sin(x)/x</code> (use "Plot")</li>
</ul>
</p>
"""

def safe_sympify(expr):
    # convert caret ^ to ** if the user used it
    expr = expr.replace('^', '**')
    # use sympy.sympify with common transforms
    try:
        # allow implicit multiplication and standard transforms
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                implicit_multiplication_application)
        transformations = standard_transformations + (implicit_multiplication_application,)
        return parse_expr(expr, transformations=transformations, evaluate=True)
    except Exception as e:
        raise ValueError(f"Could not parse expression: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    plot_available = False
    expr = ''
    op = 'simplify'
    var = 'x'
    set_prec = 15

    if request.method == 'POST':
        expr = request.form.get('expr','').strip()
        op = request.form.get('op','simplify')
        var = request.form.get('var','x').strip() or 'x'
        try:
            set_prec = max(1, int(request.form.get('set_prec', set_prec)))
        except:
            set_prec = 15

        if not expr:
            error = "Please enter an expression or equation."
            return render_template_string(HTML, error=error, result=result, plot_available=plot_available,
                                          expr=expr, op=op, var=var, set_prec=set_prec)

        try:
            # Solve operations
            if op == 'simplify':
                expr_sym = safe_sympify(expr)
                result = sp.simplify(expr_sym)
            elif op == 'evaluate':
                expr_sym = safe_sympify(expr)
                # numeric evaluation: substitute common constants and use evalf
                result = expr_sym.evalf(n=set_prec)
            elif op == 'differentiate':
                expr_sym = safe_sympify(expr)
                v = sp.symbols(var)
                result = sp.diff(expr_sym, v)
            elif op == 'integrate':
                expr_sym = safe_sympify(expr)
                v = sp.symbols(var)
                result = sp.integrate(expr_sym, v)
            elif op == 'solve':
                # expect an equation or expression = 0
                if '=' in expr:
                    left, right = expr.split('=',1)
                    lhs = safe_sympify(left)
                    rhs = safe_sympify(right)
                    eq = sp.Eq(lhs, rhs)
                    v = sp.symbols(var)
                    sol = sp.solve(eq, v)
                    result = sol if sol else "No solution found (or solution is complicated)."
                else:
                    # solve expression == 0
                    expr_sym = safe_sympify(expr)
                    v = sp.symbols(var)
                    sol = sp.solve(sp.Eq(expr_sym, 0), v)
                    result = sol if sol else "No solution found (or solution is complicated)."
            elif op == 'plot':
                # We'll mark plot available and create dynamic image via /plot.png
                # Validate expression
                _ = safe_sympify(expr)  # will raise if invalid
                plot_available = True
                result = "Plot prepared below."
            else:
                error = "Unknown operation."
        except Exception as e:
            error = str(e)

    return render_template_string(HTML, error=error, result=result, plot_available=plot_available,
                                  expr=expr, op=op, var=var, set_prec=set_prec)

@app.route('/plot.png')
def plot_png():
    expr = request.args.get('expr','')
    var = request.args.get('var','x')
    if not expr:
        return "No expression", 400
    try:
        expr_sym = safe_sympify(expr)
        v = sp.symbols(var)
        f = sp.lambdify(v, expr_sym, 'numpy')
    except Exception as e:
        return f"Error parsing expression: {e}", 400

    import numpy as np
    x = np.linspace(-10, 10, 800)
    try:
        y = f(x)
    except Exception as e:
        return f"Error evaluating function for plotting: {e}", 400

    # prepare the plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, y)
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    ax.set_title(f"y = {sp.srepr(expr_sym)}")
    ax.grid(True)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

<<<<<<< HEAD
if __name__ == '__main__':
    # for development only; for production use a proper server (gunicorn)
    app.run(debug=True, port=5000)
=======
if __name__ == "__main__":
    from os import environ
    app.run(host="0.0.0.0", port=int(environ.get("PORT", 5000)))
>>>>>>> 3d7f094b209a700c22cedcdc4ae9a050ebcfb754
