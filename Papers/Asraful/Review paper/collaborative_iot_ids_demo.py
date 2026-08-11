# We import Flask so Python can make a small website.
from flask import Flask, request, render_template_string

# We import NumPy so Python can work with small groups of numbers.
import numpy as np

# We import LogisticRegression so each IoT device can learn "Safe" or "Attack".
from sklearn.linear_model import LogisticRegression

# We create our Flask website and call it app.
app = Flask(__name__)

# Device 1 keeps its own private training data here.
device1_x = np.array([[10, 0, 100], [15, 0, 120], [20, 1, 150], [80, 5, 900], [95, 8, 1200], [70, 6, 850]])

# Device 1 answers are here: 0 means Safe and 1 means Attack.
device1_y = np.array([0, 0, 0, 1, 1, 1])

# Device 2 keeps its own private training data here.
device2_x = np.array([[8, 0, 90], [18, 1, 140], [25, 0, 180], [75, 7, 950], [88, 9, 1100], [65, 5, 780]])

# Device 2 answers are here: 0 means Safe and 1 means Attack.
device2_y = np.array([0, 0, 0, 1, 1, 1])

# Device 3 keeps its own private training data here.
device3_x = np.array([[12, 0, 110], [22, 1, 160], [30, 1, 200], [72, 6, 880], [90, 10, 1300], [68, 4, 760]])

# Device 3 answers are here: 0 means Safe and 1 means Attack.
device3_y = np.array([0, 0, 0, 1, 1, 1])

# This function teaches one small AI model using one device's private data.
def train_local_model(x, y):
    # We create a simple Logistic Regression AI model.
    model = LogisticRegression(max_iter=1000)
    # We teach the model using only this device's own private data.
    model.fit(x, y)
    # We return the trained model back to the program.
    return model

# We train a private model inside Device 1.
model1 = train_local_model(device1_x, device1_y)

# We train a private model inside Device 2.
model2 = train_local_model(device2_x, device2_y)

# We train a private model inside Device 3.
model3 = train_local_model(device3_x, device3_y)

# We put the three learned weight lists together.
all_weights = np.array([model1.coef_[0], model2.coef_[0], model3.coef_[0]])

# We average the weights to make one shared global model idea.
global_weights = np.mean(all_weights, axis=0)

# We put the three learned starting numbers together.
all_biases = np.array([model1.intercept_[0], model2.intercept_[0], model3.intercept_[0]])

# We average the starting numbers too.
global_bias = np.mean(all_biases)

# This function turns the global model's number into an attack chance.
def global_attack_probability(features):
    # We calculate one score using the shared global weights.
    score = np.dot(features, global_weights) + global_bias
    # We change the score into a value between 0 and 1.
    probability = 1 / (1 + np.exp(-score))
    # We return the attack chance.
    return probability

# This is the complete HTML page shown in the browser.
HTML = """
<!DOCTYPE html>
<html>
<head>
    <!-- This tells the browser to use normal text encoding. -->
    <meta charset="UTF-8">

    <!-- This is the name shown on the browser tab. -->
    <title>Simple Collaborative IoT Intrusion Detection Demo</title>

    <!-- This CSS makes the page clean and easy to read. -->
    <style>
        /* This changes the whole page font and background. */
        body { font-family: Arial, sans-serif; background: #f4f7fb; padding: 30px; }

        /* This white box holds our main website. */
        .box { max-width: 850px; margin: auto; background: white; padding: 25px; border-radius: 14px; }

        /* This makes the title easy to see. */
        h1 { text-align: center; }

        /* This makes each input box neat. */
        input { width: 100%; padding: 10px; margin: 6px 0 14px 0; box-sizing: border-box; }

        /* This makes the button easy to see and click. */
        button { padding: 12px 18px; cursor: pointer; }

        /* This makes the result area stand out. */
        .result { margin-top: 20px; padding: 18px; background: #eef3f8; border-radius: 10px; }

        /* This gives small information cards a simple layout. */
        .devices { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 20px; }

        /* This styles each device card. */
        .device { padding: 12px; background: #fafafa; border: 1px solid #ddd; border-radius: 10px; }
    </style>
</head>
<body>
    <!-- This is the big white box in the middle of the page. -->
    <div class="box">

        <!-- This is the main heading. -->
        <h1>Collaborative IoT Intrusion Detection</h1>

        <!-- This explains the project in very easy language. -->
        <p>
            Imagine 3 smart devices learning how to detect a cyber attack.
            Each device keeps its own private data.
            It sends only what it learned to make one shared model.
        </p>

        <!-- These cards show the three pretend IoT devices. -->
        <div class="devices">

            <!-- This is Device 1. -->
            <div class="device"><b>Device 1</b><br>Trains with its own data.</div>

            <!-- This is Device 2. -->
            <div class="device"><b>Device 2</b><br>Trains with its own data.</div>

            <!-- This is Device 3. -->
            <div class="device"><b>Device 3</b><br>Trains with its own data.</div>
        </div>

        <!-- This line explains what the server does. -->
        <p><b>Shared Server:</b> It averages what the devices learned. Raw private data is not sent here.</p>

        <!-- This form sends the user's test numbers to Python. -->
        <form method="POST">

            <!-- This asks for packets per second. -->
            <label>Packets per second:</label>

            <!-- This is the packets-per-second input box. -->
            <input type="number" name="packets" value="{{ packets }}" required>

            <!-- This asks for failed login count. -->
            <label>Failed login attempts:</label>

            <!-- This is the failed-login input box. -->
            <input type="number" name="failed" value="{{ failed }}" required>

            <!-- This asks for data size. -->
            <label>Data size in KB:</label>

            <!-- This is the data-size input box. -->
            <input type="number" name="size" value="{{ size }}" required>

            <!-- This button asks Python to check the traffic. -->
            <button type="submit">Check Traffic</button>
        </form>

        <!-- We only show this result box after the user presses the button. -->
        {% if result %}

        <!-- This box shows the final answer. -->
        <div class="result">

            <!-- This shows the final Safe or Attack result. -->
            <h2>Final Result: {{ result }}</h2>

            <!-- This shows the global model's attack chance. -->
            <p>Shared model attack chance: <b>{{ probability }}%</b></p>

            <!-- This shows Device 1's own opinion. -->
            <p>Device 1 says: <b>{{ d1 }}</b></p>

            <!-- This shows Device 2's own opinion. -->
            <p>Device 2 says: <b>{{ d2 }}</b></p>

            <!-- This shows Device 3's own opinion. -->
            <p>Device 3 says: <b>{{ d3 }}</b></p>

            <!-- This reminds the user why this is collaborative. -->
            <p>The devices learned separately, but their learned model information was combined.</p>
        </div>

        <!-- This closes the "show result only if available" rule. -->
        {% endif %}
    </div>
</body>
</html>
"""

# This tells Flask what to do when someone opens the main page.
@app.route("/", methods=["GET", "POST"])
def home():
    # We start with empty default values.
    packets = ""
    # We start with an empty failed-login value.
    failed = ""
    # We start with an empty data-size value.
    size = ""
    # We start with no final result.
    result = None
    # We start with no probability result.
    probability = None
    # We start with no Device 1 result.
    d1 = None
    # We start with no Device 2 result.
    d2 = None
    # We start with no Device 3 result.
    d3 = None

    # This runs only when the user presses the "Check Traffic" button.
    if request.method == "POST":
        # We read the packets number from the webpage.
        packets = float(request.form["packets"])
        # We read the failed-login number from the webpage.
        failed = float(request.form["failed"])
        # We read the data-size number from the webpage.
        size = float(request.form["size"])

        # We put the three user numbers into one small list for the AI.
        test_data = np.array([[packets, failed, size]])

        # Device 1 checks the traffic with its own model.
        d1_number = model1.predict(test_data)[0]
        # Device 2 checks the traffic with its own model.
        d2_number = model2.predict(test_data)[0]
        # Device 3 checks the traffic with its own model.
        d3_number = model3.predict(test_data)[0]

        # We turn Device 1's number into an easy word.
        d1 = "ATTACK" if d1_number == 1 else "SAFE"
        # We turn Device 2's number into an easy word.
        d2 = "ATTACK" if d2_number == 1 else "SAFE"
        # We turn Device 3's number into an easy word.
        d3 = "ATTACK" if d3_number == 1 else "SAFE"

        # We ask the shared global model for the attack probability.
        probability_value = global_attack_probability(test_data[0])
        # We change the probability into a percentage with 2 decimal places.
        probability = round(probability_value * 100, 2)

        # If the attack chance is 50 percent or more, we call it an attack.
        result = "ATTACK DETECTED" if probability_value >= 0.5 else "SAFE TRAFFIC"

    # We send all values to the HTML page so the browser can show them.
    return render_template_string(
        HTML,
        packets=packets,
        failed=failed,
        size=size,
        result=result,
        probability=probability,
        d1=d1,
        d2=d2,
        d3=d3
    )

# This checks whether we started this file directly.
if __name__ == "__main__":
    # This starts the website on the computer.
    app.run(debug=True)
