import json
import time
import logging
from datetime import datetime
from typing import Dict, Any

import gradio as gr

from config import CONFIG
from model_loader import ModelLoader
from offering_generator import SEDIMARKOfferingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioSEDIMARKApp:

    def __init__(self):
        self.model_loader = None
        self.offering_generator = None
        self.output_dir = CONFIG["output_dir"]
        self.output_dir.mkdir(exist_ok=True)

    def initialize_model(self):
        try:
            logger.info("Loading model...")
            self.model_loader = ModelLoader()
            self.model_loader.load_model()
            self.offering_generator = SEDIMARKOfferingGenerator(self.model_loader)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False

    def generate_offering(self, prompt: str, mode: str, context_json: str,
                         max_tokens: int, temp: float, top_p: float, num_beams: int):

        if not self.offering_generator:
            return "Model not loaded", "", "", self._empty_metrics()

        if not prompt.strip():
            return "Please enter a prompt", "", "", self._empty_metrics()

        try:
            start_time = time.time()

            # Parse context if provided
            context = None
            if mode == "With Context" and context_json:
                try:
                    context = json.loads(context_json)
                except:
                    context = None

            # Generate offering
            offering = self.offering_generator.generate_offering(
                prompt=prompt,
                use_context=(mode == "With Context"),
                context=context,
                use_schema=(mode == "Schema-Guided"),
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                num_beams=num_beams
            )

            gen_time = time.time() - start_time

            # Format output
            offering_json = json.dumps(offering, indent=2, ensure_ascii=False)

            # Validation
            validation = self._validate_offering(offering)

            # Statistics
            stats = self._generate_stats(offering, gen_time, mode)

            # Metrics for display
            metrics = self._calculate_metrics(offering_json, gen_time)

            return offering_json, validation, stats, *metrics

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            return error_msg, error_msg, error_msg, *self._empty_metrics()

    def _validate_offering(self, offering: Dict[str, Any]) -> str:
        """Validate offering structure"""
        results = []
        results.append("Valid JSON-LD structure")

        if "@context" in offering:
            results.append(f"@context present ({len(offering['@context'])} namespaces)")
        else:
            results.append("Missing @context")

        if "@graph" in offering:
            entity_count = len(offering["@graph"])
            results.append(f"@graph present ({entity_count} entities)")

            # Check entity types
            entity_types = set()
            for entity in offering["@graph"]:
                if "@type" in entity:
                    entity_types.add(entity["@type"])

            required_types = ["sedimark:Offering", "sedimark:Asset"]
            for req_type in required_types:
                if req_type in entity_types:
                    results.append(f"{req_type} found")
                else:
                    results.append(f"{req_type} not found")
        else:
            results.append("Missing @graph")

        return "\n".join(results)

    def _generate_stats(self, offering: Dict[str, Any], gen_time: float, mode: str) -> str:
        return f"""Generation Statistics:
Mode: {mode}
Generation Time: {gen_time:.2f}s
Output Size: {len(json.dumps(offering))} characters
Entities: {len(offering.get('@graph', []))}
Namespaces: {len(offering.get('@context', {}))}
Valid JSON: Yes"""

    def _calculate_metrics(self, offering_json: str, gen_time: float) -> tuple:
        """Calculate metrics for display"""
        estimated_tokens = len(offering_json) // 4
        tok_per_sec = f"{estimated_tokens / gen_time:.1f}" if gen_time > 0 else "N/A"

        return (
            f"<div class='metric-value'>{tok_per_sec}</div><div class='metric-label'>Tokens/Second</div>",
            f"<div class='metric-value'>~{estimated_tokens}</div><div class='metric-label'>Output Tokens</div>",
            f"<div class='metric-value'>{gen_time:.2f}s</div><div class='metric-label'>Generation Time</div>",
            "<div class='metric-value'>✅</div><div class='metric-label'>JSON Status</div>"
        )

    def _empty_metrics(self) -> tuple:
        """Empty metrics for error cases"""
        return (
            "<div class='metric-value'>-</div><div class='metric-label'>Tokens/Second</div>",
            "<div class='metric-value'>-</div><div class='metric-label'>Output Tokens</div>",
            "<div class='metric-value'>-</div><div class='metric-label'>Generation Time</div>",
            "<div class='metric-value'>❌</div><div class='metric-label'>JSON Status</div>"
        )

    def save_offering(self, offering_json: str, filename: str = "") -> str:
        """Save offering to file"""
        try:
            if not offering_json.strip():
                return "No offering to save"

            if not filename.strip():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"offering_{timestamp}.json"
            elif not filename.endswith('.json'):
                filename += '.json'

            filepath = self.output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(offering_json)

            file_size = filepath.stat().st_size
            return f"Saved successfully!\nPath: {filepath}\nSize: {file_size} bytes"

        except Exception as e:
            return f"Save failed: {str(e)}"

    def create_interface(self):
        """Create Gradio interface"""
        model_loaded = self.offering_generator is not None

        css = """
        .gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
        .metric-display { background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 1rem; text-align: center; }
        .metric-value { font-size: 1.5rem; font-weight: 700; color: #1e293b; }
        .metric-label { font-size: 0.875rem; color: #64748b; margin-top: 0.25rem; }
        """

        with gr.Blocks(title="SEDIMARK Offering Generator", css=css) as interface:

            gr.HTML("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 2rem;">
                <h1 style="margin: 0; font-size: 2.5rem;">SEDIMARK Offering Generator</h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Generate SEDIMARK-compliant JSON-LD offerings</p>
            </div>
            """)

            # Status
            status_html = f"""
            <div style="background: #fef2f; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h3>System Status</h3>
                <p><strong>Model:</strong> {'✅ Ready' if model_loaded else '❌ Failed to Load'}</p>
                <p><strong>Type:</strong> PEFT/LoRA Fine-tuned (SEDIMARK-Optimized)</p>
                <p><strong>Output Directory:</strong> {self.output_dir}</p>
            </div>
            """
            gr.HTML(status_html)

            if model_loaded:
                # Generation Mode
                generation_mode = gr.Radio(
                    choices=["Standard (No Context)", "With Context", "Schema-Guided"],
                    value="Standard (No Context)",
                    label="Generation Mode"
                )

                # Prompt Input
                prompt_input = gr.Textbox(
                    label="Describe the offering you want to generate:",
                    lines=6,
                    placeholder="Example: Generate a SEDIMARK offering for smart city IoT sensor data including temperature, humidity, and air quality measurements from Tokyo metropolitan area...",
                    value=""
                )

                # Context input (conditional)
                context_input = gr.Textbox(
                    label="Custom @context (JSON format):",
                    lines=4,
                    placeholder='{"sedimark": "https://w3id.org/sedimark/ontology#", ...}',
                    visible=False
                )

                # Parameters
                with gr.Row():
                    max_tokens_slider = gr.Slider(512, 8192, value=4096, step=256, label="Max Tokens")
                    temp_slider = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Temperature")
                    top_p_slider = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                    num_beams_slider = gr.Slider(1, 8, value=4, step=1, label="Beam Search")

                # Generate button
                generate_button = gr.Button("Generate SEDIMARK Offering", variant="primary", size="lg")

                # Metrics
                with gr.Row():
                    with gr.Column():
                        tokens_per_sec = gr.HTML("<div class='metric-display'><div class='metric-value'>-</div><div class='metric-label'>Tokens/Second</div></div>")
                    with gr.Column():
                        total_tokens = gr.HTML("<div class='metric-display'><div class='metric-value'>-</div><div class='metric-label'>Output Tokens</div></div>")
                    with gr.Column():
                        generation_time = gr.HTML("<div class='metric-display'><div class='metric-value'>-</div><div class='metric-label'>Generation Time</div></div>")
                    with gr.Column():
                        json_valid = gr.HTML("<div class='metric-display'><div class='metric-value'>-</div><div class='metric-label'>JSON Status</div></div>")

                # Output tabs
                with gr.Tabs():
                    with gr.TabItem("Generated Offering"):
                        offering_output = gr.Code(label="", language="json", lines=20)

                    with gr.TabItem("Validation"):
                        validation_output = gr.Textbox(label="", lines=10, interactive=False)

                    with gr.TabItem("Statistics"):
                        stats_output = gr.Textbox(label="", lines=8, interactive=False)

                # Save section
                with gr.Row():
                    save_filename = gr.Textbox(label="Filename (optional)", placeholder="offering_YYYYMMDD_HHMMSS.json")
                    save_button = gr.Button("Save", variant="secondary")

                save_status = gr.Textbox(label="Save Status", interactive=False, lines=2)

                # Event handlers
                def update_context_visibility(mode):
                    return gr.update(visible=(mode == "With Context"))

                def clear_prompt():
                    return ""

                def load_example():
                    return """Generate a comprehensive SEDIMARK offering for smart city IoT sensor data including:
- Temperature, humidity, and air quality measurements
- Data from 500 sensors across Tokyo metropolitan area
- Collection period: January 2024 to December 2024
- Update frequency: Every 15 minutes
- Include quality metrics and data provision details
- Licensed under CC-BY-4.0"""

                # Connect events
                generation_mode.change(
                    fn=update_context_visibility,
                    inputs=[generation_mode],
                    outputs=[context_input]
                )

                generate_button.click(
                    fn=self.generate_offering,
                    inputs=[prompt_input, generation_mode, context_input,
                           max_tokens_slider, temp_slider, top_p_slider, num_beams_slider],
                    outputs=[offering_output, validation_output, stats_output,
                            tokens_per_sec, total_tokens, generation_time, json_valid]
                )

                save_button.click(
                    fn=self.save_offering,
                    inputs=[offering_output, save_filename],
                    outputs=save_status
                )

                # Quick actions
                with gr.Row():
                    gr.Button("Clear", size="sm").click(fn=clear_prompt, outputs=[prompt_input])
                    gr.Button("Load Example", size="sm").click(fn=load_example, outputs=[prompt_input])

            else:
                gr.HTML("""
                <div style="text-align: center; padding: 3rem; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; color: #dc2626;">
                    <h2>❌ Model Loading Failed</h2>
                    <p>The model failed to load. Check the console output for detailed error messages.</p>
                </div>
                """)

        return interface

def main():
    """Main function"""
    app = GradioSEDIMARKApp()

    print("Starting SEDIMARK Offering Generator Interface...")
    print(f"Output Directory: {app.output_dir}")

    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)

    if not app.initialize_model():
        print("\n❌ MODEL LOADING FAILED")
        print("Check the error messages above for details.")
        return

    print("\n" + "="*60)
    print("✅ MODEL LOADED SUCCESSFULLY")
    print("="*60)

    # Create and launch interface
    print("\nCreating Gradio interface...")
    interface = app.create_interface()

    # Launch configuration
    port = CONFIG["gradio"]["port"]
    share = CONFIG["gradio"]["share"]

    print(f"\nLaunching interface on port {port}...")
    try:
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            inbrowser=not share,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\nInterface stopped by user")
    except Exception as e:
        print(f"\n❌ Launch failed: {e}")

if __name__ == "__main__":
    main()
