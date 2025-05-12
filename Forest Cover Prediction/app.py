import gradio as gr
import joblib
import numpy as np

model = joblib.load("forest_cover_model.pkl")

label_map = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

def predict_cover(elevation, aspect, slope, hor_dist_hydro, vert_dist_hydro,
                  hor_dist_road, hor_dist_fire, hillshade_9am, hillshade_noon, hillshade_3pm,
                  wilderness_choice, soil_choice):
    

    wilderness_vector = [0] * 4
    if wilderness_choice:
        idx = int(wilderness_choice.split()[-1]) - 1
        wilderness_vector[idx] = 1

    soil_vector = [0] * 40
    if soil_choice:
        idx = int(soil_choice.split()[-1]) - 1
        soil_vector[idx] = 1

    input_data = np.array([ 
        elevation, aspect, slope, hor_dist_hydro, vert_dist_hydro,
        hor_dist_road, hor_dist_fire, hillshade_9am, hillshade_noon, hillshade_3pm
    ] + wilderness_vector + soil_vector).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    return f"Predicted Cover Type: {label_map[prediction]}"

with gr.Blocks(title="Forest Cover Type Prediction", theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🌲 Forest Cover Type Prediction")
    gr.Markdown("### Predict the forest cover type for a 30m x 30m patch of land based on cartographic variables")

    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column():
                elevation = gr.Slider(0, 4000, value=2500, label="Elevation (m)")
                aspect = gr.Slider(0, 360, value=180, label="Aspect (azimuth degrees)")
                slope = gr.Slider(0, 60, value=15, label="Slope (degrees)")
                hor_dist_hydro = gr.Slider(0, 1500, value=300, label="Horizontal Distance to Hydrology (m)")
                vert_dist_hydro = gr.Slider(-500, 700, value=50, label="Vertical Distance to Hydrology (m)")
                hor_dist_road = gr.Slider(0, 8000, value=1000, label="Horizontal Distance to Roadways (m)")
                hillshade_9am = gr.Slider(0, 255, value=150, label="Hillshade 9am")
                hillshade_noon = gr.Slider(0, 255, value=200, label="Hillshade Noon")
                hillshade_3pm = gr.Slider(0, 255, value=150, label="Hillshade 3pm")

            with gr.Column():
                hor_dist_fire = gr.Slider(0, 8000, value=1500, label="Horizontal Distance to Fire Points (m)")
                
                gr.Markdown("---")
                
                wilderness = gr.Radio(choices=[f"Wilderness Area {i+1}" for i in range(4)], label="Wilderness Area (Select One)")
                
                gr.Markdown("---")
                # gr.Markdown("#### Soil Type (Select One)")

                soil = gr.Radio(
                    choices=[f"Soil Type {i+1}" for i in range(40)],
                    label="Soil Type (Select One)",
                    value="Soil Type 1",
                    interactive=True,
                    container=True
                )

        submit_btn = gr.Button("Predict Forest Cover Type")
        output = gr.Textbox(label="Prediction Result")

        all_inputs = [
            elevation, aspect, slope, hor_dist_hydro, vert_dist_hydro,
            hor_dist_road, hor_dist_fire, hillshade_9am, hillshade_noon, hillshade_3pm,
            wilderness, soil
        ]

        submit_btn.click(fn=predict_cover, inputs=all_inputs, outputs=output)

    with gr.Tab("About"):
        gr.Markdown(""" 
        # 📘 About This Project
        ## Forest Cover Type Prediction
        This application predicts the type of forest cover using cartographic variables for a 30m x 30m patch of land in the Roosevelt National Forest of northern Colorado.
            
        ### Forest Cover Types (target classes):
        1. Spruce/Fir
        2. Lodgepole Pine
        3. Ponderosa Pine
        4. Cottonwood/Willow
        5. Aspen
        6. Douglas-fir
        7. Krummholz
            
        ### Feature Information:
        - **Elevation**: Elevation in meters
        - **Aspect**: Aspect in degrees azimuth
        - **Slope**: Slope in degrees
        - **Horizontal/Vertical Distance to Hydrology**: Distance to nearest surface water features
        - **Horizontal Distance to Roadways**: Distance to nearest roadway
        - **Hillshade indices**: Hillshade index at 9am, noon, and 3pm on summer solstice
        - **Horizontal Distance to Fire Points**: Distance to nearest wildfire ignition points
        - **Wilderness Area**: 4 binary columns for wilderness area designation
        - **Soil Type**: 40 binary columns for soil type designation
            
        This application uses a Random Forest classifier trained on the forest cover type dataset.
        """)
        gr.Markdown("### Created by Aritra Mukherjee")
app.launch()
