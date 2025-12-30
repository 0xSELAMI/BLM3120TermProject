import gradio as gr
import traceback
from defaults import *

import io
from contextlib import redirect_stdout, redirect_stderr

from GUI.layout_definition import layout_definition

class ForwardArgs:
    def __init__(self, user, args):
        self.invalid = False

        for name in args:
            if args[name] is None or args[name] == '':
                print(f"[ERROR] {user}(): {name} invalid")
                self.invalid = True
                break

            setattr(self, name, args[name])

class GUI:
    def __init__(self, handlers):
        self.handlers = handlers
        self.interface = self.build_ui()

    def launch(self):
        self.interface.launch()

    def build_ui(self):
        with gr.Blocks(title="BLM3120 Term Project") as project_gui:
            gr.Markdown("# BLM3120 Information Retrieval and Search Engines Term Project")

            tab_container = gr.Tabs()

            with gr.Row():
                self.log_out = gr.Textbox(
                    label="Output",
                    lines=10,
                    placeholder="Logs will appear here...",
                    interactive=False,
                    scale=9
                )

                clear_btn = gr.Button("Clear", variant="secondary", scale=1)

            clear_btn.click(lambda: '', outputs=[self.log_out])

            with tab_container:
                for sublayout in layout_definition:
                    self.render_layout(sublayout)

        return project_gui

    def generic_forward(self, handler_path, param_names, *values):
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):

            # build args dictionary
            args_dict = dict(zip(param_names, values))

            # type conversion for comma-separated weights
            for k, v in args_dict.items():
                if isinstance(v, str) and "weights" in k:
                    args_dict[k] = [float(w) for w in v.split(',') if w.strip()]

            args = ForwardArgs(handler_path, args_dict)

            if not args.invalid:
                handler = self.handlers

                for key in handler_path.split('.'):
                    handler = handler[key]

                handler(args)

            return f.getvalue()

    def render_layout(self, layout):
        with gr.Tab(layout["title"]):
            if "subtabs" in layout:
                with gr.Tabs():
                    for sub in layout["subtabs"]:
                        self.render_layout(sub) # Recursion magic
            else:
                all_inputs = []
                all_keys = []

                for section in layout.get("sections", []):
                    self.create_section(section, all_inputs, all_keys)

                btn = gr.Button(f"{layout['title']}", variant="primary")

                btn.click(
                    fn=lambda *args: self.generic_forward(layout["handler"], all_keys, *args),
                    inputs=all_inputs,
                    outputs=[self.log_out]
                )

    # create sections recursively
    def create_section(self, section, all_inputs, all_keys):
        layout = section.get("layout", "column")

        if layout == "row":
            container = gr.Row()
        elif layout == "accordion":
            container = gr.Accordion(section.get("label", "Advanced"), open=False)
        elif layout == "group":
            container = gr.Group()
        else:
            container = gr.Column()

        with container:
            for field in section.get("fields", []):
                comp = self.create_component(field)
                all_inputs.append(comp)
                all_keys.append(field["id"])

            for sub_section in section.get("sections", []):
                self.create_section(sub_section, all_inputs, all_keys)

    def create_component(self, field):
        # map data from layout definition to gradio components
        ftype = field.get("type", "text")
        label = field.get("label", "Input")
        value = field.get("value")
        info = field.get("info", "")

        if ftype == "path":
            return gr.Textbox(label=label, value=value, info=f"{info} (relative to project directory)", placeholder="path/to/file")

        if ftype == "number":
            return gr.Number(label=label, value=value, info=info)

        if ftype == "dropdown":
            return gr.Dropdown(label=label, choices=field.get("choices", [True, False]), value=value, info=info)

        # default to Textbox
        return gr.Textbox(label=label, value=value, info=info)

