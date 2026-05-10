"""Панель управления с поддержкой разных генераторов"""

from tkinter import (
    Frame,
    Label,
    Entry,
    Button,
    Scale,
    Radiobutton,
    LabelFrame,
    Checkbutton,
    StringVar,
    IntVar,
    DoubleVar,
    BooleanVar,
)
from tkinter import LEFT, RIGHT, X, Y, W, EW, NORMAL, DISABLED, HORIZONTAL
from tkinter import ttk


class ControlsPanel(Frame):
    def __init__(self, parent, on_generate, on_run, on_randomize_layout, on_apply_force):
        super().__init__(parent, bg="#2c3e50")
        self.on_generate = on_generate
        self.on_run = on_run
        self.on_randomize_layout = on_randomize_layout
        self.on_apply_force = on_apply_force

        self.current_generator = StringVar(value="Cluster")
        self._create_widgets()

    def _create_widgets(self):
        # === Generator Selection ===
        gen_frame = LabelFrame(self, text="Graph Generator", bg="#2c3e50", fg="white", font=("Arial", 10, "bold"))
        gen_frame.pack(fill=X, padx=5, pady=5)

        # Тип генератора
        Label(gen_frame, text="Type:", bg="#2c3e50", fg="white").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.gen_type = ttk.Combobox(
            gen_frame,
            textvariable=self.current_generator,
            values=["Cluster", "FastCluster", "Barabasi-Albert", "Erdos-Renyi"],
            width=15,
        )
        self.gen_type.grid(row=0, column=1, padx=5, pady=2)
        self.gen_type.bind("<<ComboboxSelected>>", self._on_generator_change)

        # Общий параметр
        Label(gen_frame, text="Vertices:", bg="#2c3e50", fg="white").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.v_entry = Entry(gen_frame, width=12, bg="#34495e", fg="white")
        self.v_entry.insert(0, "100")
        self.v_entry.grid(row=1, column=1, padx=5, pady=2)

        # === Динамические параметры (будут меняться) ===
        self.dynamic_frame = Frame(gen_frame, bg="#2c3e50")
        self.dynamic_frame.grid(row=2, column=0, columnspan=2, sticky=EW, pady=5)

        # === Кнопка генерации ===
        Button(gen_frame, text="Generate Graph", command=self._generate, bg="#3498db", fg="white").grid(
            row=3, column=0, columnspan=2, pady=10, sticky=EW
        )

        gen_frame.columnconfigure(1, weight=1)

        # Инициализируем параметры для генератора по умолчанию
        self._on_generator_change()

        # === Algorithm Settings ===
        a_frame = LabelFrame(self, text="Algorithm", bg="#2c3e50", fg="white", font=("Arial", 10, "bold"))
        a_frame.pack(fill=X, padx=5, pady=5)

        self.algo_var = StringVar(value="Multilevel")
        Radiobutton(
            a_frame,
            text="Kernighan-Lin",
            variable=self.algo_var,
            value="KL",
            bg="#2c3e50",
            fg="white",
            selectcolor="#2c3e50",
        ).pack(anchor=W, padx=10, pady=2)
        Radiobutton(
            a_frame,
            text="Multilevel",
            variable=self.algo_var,
            value="Multilevel",
            bg="#2c3e50",
            fg="white",
            selectcolor="#2c3e50",
        ).pack(anchor=W, padx=10, pady=2)

        Label(a_frame, text="Max passes (KL):", bg="#2c3e50", fg="white").pack(anchor=W, padx=10, pady=2)
        self.passes_entry = Entry(a_frame, width=10, bg="#34495e", fg="white")
        self.passes_entry.insert(0, "20")
        self.passes_entry.pack(anchor=W, padx=10, pady=2)

        Label(a_frame, text="Min coarse vertices:", bg="#2c3e50", fg="white").pack(anchor=W, padx=10, pady=2)
        self.min_coarse_entry = Entry(a_frame, width=10, bg="#34495e", fg="white")
        self.min_coarse_entry.insert(0, "20")
        self.min_coarse_entry.pack(anchor=W, padx=10, pady=2)

        Label(a_frame, text="Balance (0.3-0.7):", bg="#2c3e50", fg="white").pack(anchor=W, padx=10, pady=2)
        self.balance_var = DoubleVar(value=0.5)
        Scale(
            a_frame,
            from_=0.3,
            to=0.7,
            resolution=0.01,
            variable=self.balance_var,
            orient=HORIZONTAL,
            bg="#2c3e50",
            length=200,
        ).pack(fill=X, padx=10, pady=2)

        Button(
            a_frame, text="Run Algorithm", command=self._run, bg="#e74c3c", fg="white", font=("Arial", 10, "bold")
        ).pack(fill=X, padx=10, pady=10)

        # === View Controls ===
        v_frame = LabelFrame(self, text="View", bg="#2c3e50", fg="white", font=("Arial", 10, "bold"))
        v_frame.pack(fill=X, padx=5, pady=5)

        Button(v_frame, text="Randomize Layout", command=self.on_randomize_layout, bg="#f39c12", fg="white").pack(
            fill=X, padx=10, pady=2
        )
        Button(v_frame, text="Apply Force Layout", command=self.on_apply_force, bg="#9b59b6", fg="white").pack(
            fill=X, padx=10, pady=2
        )

        self.stages_btn = Button(
            v_frame, text="Show Coarsening Stages", command=None, bg="#95a5a6", fg="white", state=DISABLED
        )
        self.stages_btn.pack(fill=X, padx=10, pady=2)

    def _on_generator_change(self, event=None):
        # Очищаем динамическую область
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        gen_type = self.current_generator.get()

        if gen_type == "Cluster":
            self._add_cluster_params()
        elif gen_type == "FastCluster":
            self._add_fast_cluster_params()
        elif gen_type == "Barabasi-Albert":
            self._add_barabasi_params()
        elif gen_type == "Erdos-Renyi":
            self._add_erdos_renyi_params()

    def _add_cluster_params(self):
        """Параметры для ClusterGraphGenerator"""
        row = 0

        Label(self.dynamic_frame, text="Clusters:", bg="#2c3e50", fg="white").grid(
            row=row, column=0, sticky=W, padx=5, pady=2
        )
        self.c_entry = Entry(self.dynamic_frame, width=10, bg="#34495e", fg="white")
        self.c_entry.insert(0, "5")
        self.c_entry.grid(row=row, column=1, padx=5, pady=2)

        row += 1
        Label(self.dynamic_frame, text="Intra prob:", bg="#2c3e50", fg="white").grid(
            row=row, column=0, sticky=W, padx=5, pady=2
        )
        self.intra_var = DoubleVar(value=0.8)
        Scale(
            self.dynamic_frame,
            from_=0.1,
            to=1.0,
            resolution=0.01,
            variable=self.intra_var,
            orient=HORIZONTAL,
            bg="#2c3e50",
            length=150,
        ).grid(row=row, column=1, padx=5, pady=2)
        

        self._add_weight_params(row + 1)

    def _add_fast_cluster_params(self):
        """Параметры для FastClusterGenerator"""
        row = 0

        Label(self.dynamic_frame, text="Clusters:", bg="#2c3e50", fg="white").grid(
            row=row, column=0, sticky=W, padx=5, pady=2
        )
        self.fc_clusters = Entry(self.dynamic_frame, width=10, bg="#34495e", fg="white")
        self.fc_clusters.insert(0, "5")
        self.fc_clusters.grid(row=row, column=1, padx=5, pady=2)

        row += 1
        Label(self.dynamic_frame, text="Target edges:", bg="#2c3e50", fg="white").grid(
            row=row, column=0, sticky=W, padx=5, pady=2
        )
        self.fc_edges = Entry(self.dynamic_frame, width=10, bg="#34495e", fg="white")
        self.fc_edges.insert(0, "500")
        self.fc_edges.grid(row=row, column=1, padx=5, pady=2)

        row += 1
        Label(self.dynamic_frame, text="Intra ratio:", bg="#2c3e50", fg="white").grid(
            row=row, column=0, sticky=W, padx=5, pady=2
        )
        self.fc_intra_ratio = DoubleVar(value=0.7)
        Scale(
            self.dynamic_frame,
            from_=0.1,
            to=1.0,
            resolution=0.01,
            variable=self.fc_intra_ratio,
            orient=HORIZONTAL,
            bg="#2c3e50",
            length=150,
        ).grid(row=row, column=1, padx=5, pady=2)

        self._add_weight_params(row + 1)

    def _add_barabasi_params(self):
        """Параметры для BarabasiAlbertGenerator"""
        row = 0

        Label(self.dynamic_frame, text="m0 (initial vertices):", bg="#2c3e50", fg="white").grid(
            row=row, column=0, sticky=W, padx=5, pady=2
        )
        self.ba_m0 = Entry(self.dynamic_frame, width=10, bg="#34495e", fg="white")
        self.ba_m0.insert(0, "10")
        self.ba_m0.grid(row=row, column=1, padx=5, pady=2)

        row += 1
        Label(self.dynamic_frame, text="m (edges per new vertex):", bg="#2c3e50", fg="white").grid(
            row=row, column=0, sticky=W, padx=5, pady=2
        )
        self.ba_m = Entry(self.dynamic_frame, width=10, bg="#34495e", fg="white")
        self.ba_m.insert(0, "3")
        self.ba_m.grid(row=row, column=1, padx=5, pady=2)

        self._add_weight_params(row + 1)

    def _add_erdos_renyi_params(self):
        """Параметры для ErdosRenyiGenerator"""
        row = 0

        Label(self.dynamic_frame, text="Edge probability:", bg="#2c3e50", fg="white").grid(
            row=row, column=0, sticky=W, padx=5, pady=2
        )
        self.er_p = DoubleVar(value=0.05)
        Scale(
            self.dynamic_frame,
            from_=0.01,
            to=0.2,
            resolution=0.005,
            variable=self.er_p,
            orient=HORIZONTAL,
            bg="#2c3e50",
            length=150,
        ).grid(row=row, column=1, padx=5, pady=2)

        self._add_weight_params(row + 1)

    def _add_weight_params(self, row):
        """Общие параметры весов"""
        Label(self.dynamic_frame, text="Vertex weight:", bg="#2c3e50", fg="white").grid(
            row=row, column=0, sticky=W, padx=5, pady=2
        )
        vw_frame = Frame(self.dynamic_frame, bg="#2c3e50")
        vw_frame.grid(row=row, column=1, sticky=W, padx=5, pady=2)
        self.vw_min = Entry(vw_frame, width=5, bg="#34495e", fg="white")
        self.vw_min.insert(0, "1")
        self.vw_min.pack(side=LEFT)
        Label(vw_frame, text="-", bg="#2c3e50", fg="white").pack(side=LEFT, padx=2)
        self.vw_max = Entry(vw_frame, width=5, bg="#34495e", fg="white")
        self.vw_max.insert(0, "10")
        self.vw_max.pack(side=LEFT)

        row += 1
        Label(self.dynamic_frame, text="Edge weight:", bg="#2c3e50", fg="white").grid(
            row=row, column=0, sticky=W, padx=5, pady=2
        )
        ew_frame = Frame(self.dynamic_frame, bg="#2c3e50")
        ew_frame.grid(row=row, column=1, sticky=W, padx=5, pady=2)
        self.ew_min = Entry(ew_frame, width=5, bg="#34495e", fg="white")
        self.ew_min.insert(0, "1")
        self.ew_min.pack(side=LEFT)
        Label(ew_frame, text="-", bg="#2c3e50", fg="white").pack(side=LEFT, padx=2)
        self.ew_max = Entry(ew_frame, width=5, bg="#34495e", fg="white")
        self.ew_max.insert(0, "5")
        self.ew_max.pack(side=LEFT)

    def _get_weight_params(self):
        return {
            "vw_min": int(self.vw_min.get()),
            "vw_max": int(self.vw_max.get()),
            "ew_min": int(self.ew_min.get()),
            "ew_max": int(self.ew_max.get()),
        }

    def _get_generator_params(self):
        """Собирает параметры для текущего генератора"""
        gen_type = self.current_generator.get()
        n = int(self.v_entry.get())
        weights = self._get_weight_params()

        base_params = {"type": gen_type, "vertices": n, **weights}

        if gen_type == "Cluster":
            base_params.update(
                {
                    "clusters": int(self.c_entry.get()),
                    "intra_prob": self.intra_var.get(),
                }
            )
        elif gen_type == "FastCluster":
            base_params.update(
                {
                    "clusters": int(self.fc_clusters.get()),
                    "target_edges": int(self.fc_edges.get()),
                    "intra_ratio": self.fc_intra_ratio.get(),
                }
            )
        elif gen_type == "Barabasi-Albert":
            base_params.update({"m0": int(self.ba_m0.get()), "m": int(self.ba_m.get())})
        elif gen_type == "Erdos-Renyi":
            base_params.update({"p": self.er_p.get()})

        return base_params

    def _generate(self):
        params = self._get_generator_params()
        self.on_generate(params)

    def _run(self):
        params = {
            "algorithm": self.algo_var.get(),
            "max_passes": int(self.passes_entry.get()),
            "min_coarse": int(self.min_coarse_entry.get()),
            "balance": self.balance_var.get(),
        }
        self.on_run(params)

    def get_stages_button(self):
        return self.stages_btn

    def set_stages_enabled(self, enabled):
        state = NORMAL if enabled else DISABLED
        self.stages_btn.config(state=state)
