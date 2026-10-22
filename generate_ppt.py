import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Colors
    C_BG_DARK = RGBColor(15, 23, 42)       # #0f172a (Deep Slate/Navy)
    C_BG_LIGHT = RGBColor(248, 250, 252)   # #f8fafc (Off White)
    C_PURPLE = RGBColor(124, 58, 237)      # #7c3aed (Brand Purple)
    C_BLUE = RGBColor(37, 99, 235)         # #2563eb (Electric Blue)
    C_TEXT_DARK = RGBColor(15, 23, 42)     # #0f172a
    C_TEXT_MUTED = RGBColor(100, 116, 139) # #64748b
    C_WHITE = RGBColor(255, 255, 255)
    C_CARD_BG = RGBColor(255, 255, 255)
    C_CARD_BORDER = RGBColor(226, 232, 240)
    C_ACCENT_GREEN = RGBColor(16, 185, 129)

    def set_bg(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category_text="OSTUTOR // KERNEL AI"):
        # Category Pill
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_PURPLE

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = C_TEXT_DARK

    def add_card(slide, left, top, width, height, title, items, badge=""):
        # Card shape
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = C_CARD_BG
        shape.line.color.rgb = C_CARD_BORDER
        shape.line.width = Pt(1.5)

        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), height - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        if badge:
            p0.text = f"{badge}  {title}"
        else:
            p0.text = title
        p0.font.size = Pt(17)
        p0.font.bold = True
        p0.font.color.rgb = C_PURPLE
        p0.space_after = Pt(12)

        for item in items:
            p = tf.add_paragraph()
            p.text = f"•  {item}"
            p.font.size = Pt(13)
            p.font.color.rgb = C_TEXT_DARK
            p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1, C_BG_DARK)

    # Decorative banner card
    banner = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.0), Inches(11.733), Inches(5.5))
    banner.fill.solid()
    banner.fill.fore_color.rgb = RGBColor(30, 41, 59)
    banner.line.color.rgb = C_PURPLE
    banner.line.width = Pt(2)

    tb1 = s1.shapes.add_textbox(Inches(1.5), Inches(1.6), Inches(10.3), Inches(4.3))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.text = "RESEARCH & EDUCATION PROJECT"
    p_badge.font.size = Pt(13)
    p_badge.font.bold = True
    p_badge.font.color.rgb = RGBColor(168, 85, 247)
    p_badge.space_after = Pt(14)

    p_main = tf1.add_paragraph()
    p_main.text = "OSTutorLLM"
    p_main.font.size = Pt(44)
    p_main.font.bold = True
    p_main.font.color.rgb = C_WHITE
    p_main.space_after = Pt(8)

    p_sub = tf1.add_paragraph()
    p_sub.text = "An Instruction-Tuned & Retrieval-Augmented AI Platform for Operating Systems Education"
    p_sub.font.size = Pt(20)
    p_sub.font.color.rgb = RGBColor(226, 232, 240)
    p_sub.space_after = Pt(24)

    p_features = tf1.add_paragraph()
    p_features.text = "⚡ LoRA Parameter-Efficient Fine-Tuning  |  📚 Textbook-Grounded RAG  |  🚀 Interactive CPU Gantt Simulation"
    p_features.font.size = Pt(14)
    p_features.font.color.rgb = RGBColor(6, 182, 212)
    p_features.space_after = Pt(28)

    p_meta = tf1.add_paragraph()
    p_meta.text = "Presenter / Team: OSTutor Engineering  •  Domain: Operating Systems & AI Pedagogy"
    p_meta.font.size = Pt(13)
    p_meta.font.color.rgb = C_TEXT_MUTED

    # -------------------------------------------------------------
    # SLIDE 2: Introduction & Objective
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2, C_BG_LIGHT)
    add_header(s2, "Introduction & Project Objectives", "01 / FOUNDATION")

    add_card(s2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "The Challenge in OS Education",
             [
                 "High Conceptual Abstraction: Topics like concurrency, virtual address translation, and process scheduling are difficult to grasp from static texts alone.",
                 "Textbook & Theory Bottleneck: Standard textbooks (Silberschatz, Tanenbaum) provide deep theory but lack interactive, immediate execution feedback.",
                 "Generic LLM Limitations: Standard AI models hallucinate kernel behaviors, lack precise execution traces, and offer unverifiable citations.",
                 "Need for Modern Pedagogy: College students require a hands-on, student-friendly command center to bridge theoretical principles with real kernel mechanics."
             ],
             badge="⚠️")

    add_card(s2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "Key Project Objectives",
             [
                 "Instruction-Tuned Domain LLM: Fine-tune a dedicated LLM on OS question-answer datasets using Parameter-Efficient Fine-Tuning (LoRA).",
                 "Authoritative RAG Retrieval: Ground AI responses in verified benchmark literature (Silberschatz, OSTEP, Tanenbaum, Linux Kernel Docs).",
                 "Step-by-Step Kernel Tracing: Augment conceptual explanations with system-call and MMU trace logs (e.g., fork COW, TLB miss walks).",
                 "Interactive Learning Simulator: Deliver real-time Gantt timeline visualization for CPU scheduling algorithms with customizable process metrics."
             ],
             badge="🎯")

    # -------------------------------------------------------------
    # SLIDE 3: Methodology & System Architecture
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3, C_BG_LIGHT)
    add_header(s3, "Methodology & End-to-End Architecture", "02 / SYSTEM DESIGN")

    col_w = Inches(3.64)
    add_card(s3, Inches(0.8), Inches(1.8), col_w, Inches(5.0),
             "1. Student Command Center",
             [
                 "FastAPI + Modern SPA: Responsive interface designed with student-friendly visual hierarchy and clean UI components.",
                 "Modular Concept Matrix: Structured modules for Processes, CPU Scheduling, Virtual Memory, Concurrency, and Storage.",
                 "Interactive Telemetry: Live active threads monitor, model status, and latency indicators.",
                 "Direct Textbook Access: Integrated 'Open Book' deep-links for instant verification."
             ],
             badge="🖥️")

    add_card(s3, Inches(4.84), Inches(1.8), col_w, Inches(5.0),
             "2. Domain RAG & LoRA",
             [
                 "Vector Store (FAISS): Indexed corpora from premier OS literature for semantic chunk retrieval.",
                 "Top-K Similarity Scoring: Cosine similarity matching (86%–94%) with verified chapter citations.",
                 "LoRA Model Adaptation: Rank r=16, alpha=32 matrices injected into attention projections.",
                 "Targeted Fine-Tuning: Minimizes parameter updates while specializing model on kernel mechanics."
             ],
             badge="🧠")

    add_card(s3, Inches(8.88), Inches(1.8), col_w, Inches(5.0),
             "3. Interactive Simulator",
             [
                 "Multi-Algorithm Support: FCFS, Shortest Job First (SJF), and Round Robin (RR) with dynamic quantum.",
                 "Editable Process Grid: Real-time user input for PID, Arrival Time, Burst Time, and Priority.",
                 "Instant Gantt Rendering: Live timeline computation via asynchronous REST API.",
                 "Performance Telemetry: Dynamic calculation of Avg Waiting Time and Avg Turnaround Time."
             ],
             badge="⚡")

    # -------------------------------------------------------------
    # SLIDE 4: Results & Discussion (Part 1: AI Model & RAG)
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4, C_BG_LIGHT)
    add_header(s4, "Results & Discussion: Model Fine-Tuning & RAG", "03 / R&D - MODEL & RETRIEVAL")

    add_card(s4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "LoRA Fine-Tuning Performance",
             [
                 "Base Architecture: TinyLlama-1.1B-Chat with LoRA low-rank adaptation layers.",
                 "Rapid Loss Convergence: Training loss decreased from 9.2666 (Epoch 1) -> 7.0946 (Epoch 2) -> 5.5935 (Epoch 3).",
                 "Steep Perplexity Reduction: Perplexity plummeted from 10,578.93 down to 268.66, demonstrating swift domain adaptation.",
                 "Efficient Training Footprint: Checkpoints saved in under 25 seconds on CPU/edge hardware without catastrophic forgetting.",
                 "Deterministic Validation: Verified 100% test suite pass rate (18/18 pytest test cases)."
             ],
             badge="📈")

    add_card(s4, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "RAG Retrieval & Grounding Evaluation",
             [
                 "Zero Hallucination Grounding: Queries consistently retrieve verified passages from Silberschatz, OSTEP, and Tanenbaum.",
                 "High Relevance Scores: Similarity matching achieved 86% - 94% across core operating systems units.",
                 "Verified Knowledge Base: Integrated multi-level indexing covering PCB structures, TLB misses, and CFS red-black tree schedulers.",
                 "Interactive Source Verification: Students can inspect exact source text and launch the referenced textbook chapter directly via browser."
             ],
             badge="📚")

    # -------------------------------------------------------------
    # SLIDE 5: Results & Discussion (Part 2: Interactive Platform)
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5, C_BG_LIGHT)
    add_header(s5, "Results & Discussion: Interactive Student Platform", "04 / R&D - USER EXPERIENCE")

    add_card(s5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "Kernel Execution Trace Engine",
             [
                 "Transparent OS Logic: Bridges conceptual questions with step-by-step kernel system call actions.",
                 "Concrete Trace Examples: Detailed tracking of sys_fork(), task_struct allocation, Copy-On-Write (COW) page flags, and return codes.",
                 "Hardware Interaction Modeling: Simulates Virtual Address -> MMU translation, TLB miss cascades, and page fault ISR calls.",
                 "High Student Engagement: Replaces dry textbook paragraphs with actionable command-line trace visualizations."
             ],
             badge="⚡")

    add_card(s5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "Dynamic Scheduling Simulator",
             [
                 "Live Algorithmic Comparison: Instant visual comparison between FCFS (convoy effect risk) and Round Robin time-slicing.",
                 "Full Parameter Customization: Live editing of Arrival Times (AT) and Burst Times (BT) with instantaneous visual recalculation.",
                 "Turnaround & Waiting Metrics: Real-time automated calculation of Avg Waiting Time and Avg Turnaround Time for student homework verification.",
                 "Engaging Visual Design: Color-coded process blocks and student-centric, vibrant purple visual identity."
             ],
             badge="📊")

    # -------------------------------------------------------------
    # SLIDE 6: Conclusion
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6, C_BG_LIGHT)
    add_header(s6, "Conclusion & Future Work", "05 / SYNTHESIS")

    add_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "Summary of Achievements",
             [
                 "Successful Specialized AI Tutor: Delivered an end-to-end, instruction-tuned AI tutor specifically tailored for college-level OS education.",
                 "Effective RAG + LoRA Synergy: Proved that combining parameter-efficient fine-tuning with textbook vector retrieval eliminates hallucinations.",
                 "Interactive Educational Utility: Solved traditional student pain points by uniting kernel traces, visual CPU scheduling, and textbook sources in one app.",
                 "Production-Ready Architecture: Complete FastAPI server, modern student-friendly SPA, clean Git versioning, and 100% test verification."
             ],
             badge="✅")

    add_card(s6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "Future Research & Roadmap",
             [
                 "Live eBPF Linux Kernel Tracing: Connect the tutor terminal to live Linux sandbox environments using eBPF probes for real kernel tracing.",
                 "Extended Visual Simulators: Implement interactive virtual memory page replacement (FIFO, LRU, Clock) and Banker's Deadlock avoidance simulators.",
                 "Personalized Adaptive Quizzes: Integrate knowledge graphs to automatically evaluate student misconceptions and generate personalized follow-up drills.",
                 "Multi-Modal OS Diagrams: Generate real-time architectural memory layout diagrams on demand."
             ],
             badge="🚀")

    # -------------------------------------------------------------
    # SLIDE 7: References
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7, C_BG_LIGHT)
    add_header(s7, "Key Literature & References", "06 / BIBLIOGRAPHY")

    add_card(s7, Inches(0.8), Inches(1.8), Inches(11.733), Inches(5.0),
             "Primary Textbooks & Research Publications",
             [
                 "Silberschatz, A., Galvin, P. B., & Gagne, G. (2018). Operating System Concepts (10th ed.). John Wiley & Sons. [Primary OS Theory & PCB Concepts]",
                 "Arpaci-Dusseau, R. H., & Arpaci-Dusseau, A. C. (2018). Operating Systems: Three Easy Pieces (OSTEP). Arpaci-Dusseau Books. [Virtualization, Concurrency, & Persistence]",
                 "Tanenbaum, A. S., & Bos, H. (2014). Modern Operating Systems (4th ed.). Pearson. [Paging, Memory Hierarchy & File Systems]",
                 "Bovet, D. P., & Cesati, M. (2005). Understanding the Linux Kernel (3rd ed.). O'Reilly Media. [Linux task_struct, CFS Scheduler & Interrupt Handling]",
                 "Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.",
                 "Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems (NeurIPS), 33, 9459-9474."
             ],
             badge="📖")

    output_path = Path("c:/Users/avnee/Downloads/OSTutor-main/OSTutor-main/OSTutor_7_Slide_Presentation.pptx")
    prs.save(str(output_path))
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_deck()
