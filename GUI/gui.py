import gradio as gr
import inspect
import traceback
from defaults import *

import io
from contextlib import redirect_stdout, redirect_stderr

from GUI.layout_definition import layout_definition

import common.Logger as CommonLogger

class ForwardArgs:
    def __init__(self, user, args):
        self.invalid = False

        for name in args:
            if args[name] is None or args[name] == '':
                CommonLogger.logger.log(f"[ERROR] {user}(): {name} invalid")
                self.invalid = True
                break

            setattr(self, name, args[name])

class GUI:
    def __init__(self, handlers):
        self.elems     = {}
        self.handlers  = handlers
        self.interface = self.build_ui()

    def launch(self):
        self.interface.launch()

    def build_ui(self):
        with gr.Blocks(title="BLM3120 Term Project") as project_gui:
            gr.Markdown("# BLM3120 Information Retrieval and Search Engines Term Project")

            hide_output_group_states = {}

            tab_container = gr.Tabs()

            with gr.Row():
                self.log_out = gr.Textbox(
                    label="Output",
                    lines=10,
                    max_lines=20,
                    placeholder="Logs will appear here...",
                    interactive=False,
                    autoscroll=True,
                    scale=9
                )

                self.log_out_clear_btn = gr.Button("Clear", variant="secondary", scale=1)

            self.log_out_clear_btn.click(lambda: '', outputs=[self.log_out])

            with tab_container:
                for sublayout in layout_definition:
                    self.render_layout(sublayout, hide_output_group_states)

        return project_gui

    def generic_forward(self, handler_path, param_names, *values):
        CommonLogger.logger.clear()

        # build args dictionary
        args_dict = dict(zip(param_names, values))

        # type conversion for comma-separated weights
        for k, v in args_dict.items():
            if isinstance(v, str) and "weights" in k:
                args_dict[k] = [float(w) for w in v.split(',') if w.strip()]

        args = ForwardArgs(handler_path, args_dict)

        if not args.invalid:
            try:
                handler = self.handlers

                for key in handler_path.split('.'):
                    handler = handler[key]

                result = handler(args)

                if inspect.isgenerator(result):
                    for _ in result:
                        yield CommonLogger.logger.read_all()
                else:
                    yield CommonLogger.logger.read_all()

            except Exception:
                CommonLogger.logger.log("\n" + "="*20 + "\n[ERROR]\n" + "="*20)
                CommonLogger.logger.log(traceback.format_exc())
                yield CommonLogger.logger.read_all()
        else:
            yield CommonLogger.logger.read_all()

    def render_layout(self, layout, hide_output_group_states):
        with gr.Tab(layout["title"]) as cur_tab:
            
            if "subtabs" in layout:
                hide_output_group_states[layout["tab_id"]] = gr.State(False)

                # read from the state whether the leaf tab wants the output group invisible
                cur_tab.select(
                    fn=lambda h: (h, gr.update(visible=not h), gr.update(visible=not h)),
                    inputs=[hide_output_group_states[layout["tab_id"]]],
                    outputs=[hide_output_group_states[layout["tab_id"]], self.log_out, self.log_out_clear_btn]
                )

                with gr.Tabs():
                    for sub in layout["subtabs"]:
                        self.render_layout(sub, hide_output_group_states) # recurse
            else:

                hide_val = layout.get("hide_output_group", False)

                if '.' not in layout["handler"]:
                    cur_tab.select(
                        fn=lambda: [gr.update(visible = not hide_val)] * 2,
                        outputs=[self.log_out, self.log_out_clear_btn]
                    )
                else:

                # hide log_out and clear btn if "hide_output_group" in layout def
                    cur_tab.select(
                        fn=lambda: (hide_val, *([gr.update(visible = not hide_val)] * 2)),
                        outputs=[hide_output_group_states[layout["handler"].split('.')[0]], self.log_out, self.log_out_clear_btn]
                    )

                all_inputs = []
                all_keys = []

                target_id = layout.get("outputs_to")

                clear_btn = None
                btn = None

                if layout.get("btn_on_top", False):
                    btn = gr.Button(f"{layout['title']}", variant="primary")

                    if target_id:
                        clear_btn = gr.Button(f"Clear", variant="secondary")

                for section in layout.get("sections", []):
                    self.create_section(layout, section, all_inputs, all_keys)

                if not btn:
                    btn = gr.Button(f"{layout['title']}", variant="primary")

                    if target_id and not clr_btn:
                        clear_btn = gr.Button(f"Clear", variant="secondary")

                output_target_elem = self.log_out

                if target_id:
                    for elem_id in self.elems:
                        if layout["handler"] + target_id == elem_id:
                            output_target_elem = self.elems[elem_id]
                            break

                    clear_btn.click(lambda: '', outputs=[output_target_elem])

                btn.click(
                    fn=self.generic_forward,
                    inputs=[gr.State(layout["handler"]), gr.State(all_keys)] + all_inputs,
                    outputs=[output_target_elem]
                )

    # create sections recursively
    def create_section(self, parent_layout, section, all_inputs, all_keys):
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

                if not field.get("not_an_input", False):
                    all_inputs.append(comp)
                    all_keys.append(field["id"])

                # we keep track of all our elems,
                # also using the handler of layout to distinguish between elems with same id
                self.elems[parent_layout["handler"] + field["id"]] = comp

            for sub_section in section.get("sections", []):
                self.create_section(parent_layout, sub_section, all_inputs, all_keys) # recurse

    def create_component(self, field):
        # map data from layout definition to gradio components
        ftype = field.get("type", "text")
        label = field.get("label", "Input")
        value = field.get("value")
        info  = field.get("info", "")
        interactive = field.get("interactive", True)

        if ftype == "path":
            return gr.Textbox(label=label, value=value, info=f"{info} (relative to project directory)", placeholder="path/to/file")

        if ftype == "number":
            return gr.Number(label=label, value=value, info=info)

        if ftype == "dropdown":
            return gr.Dropdown(label=label, choices=field.get("choices", [True, False]), value=value, info=info)

        if ftype == "code":
            lines = field.get("lines", 10)
            return gr.Code(label=label, value=value, lines=lines, interactive=interactive)

        if ftype == "image":
            return gr.Image(label=label, value=value, interactive=interactive)

        if ftype == "html":
            js_on_load  = field.get("js_on_load", "")
            return gr.HTML(label=label, value=value, interactive=interactive, js_on_load=js_on_load)

        # default to Textbox
        return gr.Textbox(label=label, value=value, info=info)

