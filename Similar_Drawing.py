import os
import io
import json
import hashlib
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import faiss
import timm
from PIL import Image, ImageTk
import fitz  # PyMuPDF
from torchvision import transforms
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter import Frame, Label, Button, Canvas, Scrollbar
import sys
import threading

# Fix OpenMP library conflict - must be set before importing torch/numpy
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Configuration
MODEL_NAME = "timm/vit_small_patch14_dinov2.lvd142m"
INDEX_PATH = "pdf_embeddings.index"
METADATA_PATH = "pdf_metadata.json"
UPLOAD_DIR = "uploaded_pdfs"
IMAGES_DIR = "pdf_images"  # Directory to store small images
SIMILARITY_THRESHOLD = 1e-7
EMBED_DIM = 384
IMG_MAX_WIDTH, IMG_MAX_HEIGHT = 300, 200
SAVED_IMG_WIDTH, SAVED_IMG_HEIGHT = 300, 200  # Size for saved images

# Suppress MuPDF errors
fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)
fitz.TOOLS.reset_mupdf_warnings()

# Create directories
Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path(IMAGES_DIR).mkdir(exist_ok=True)

class ConsoleRedirect:
    """Redirect console output to tkinter Text widget"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        
    def write(self, msg):
        self.text_widget.insert(tk.END, msg)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()
        
    def flush(self):
        pass

class PDFSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Similar Drawing")
        self.root.geometry("1600x900")
        
        # Initialize model and index
        self.setup_model()
        self.load_or_create_index()
        
        # Create GUI
        self.create_widgets()
        
        # Setup console redirection
        self.setup_console_redirect()
        
    def setup_model(self):
        """Initialize the DINOv2 model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Option 1: Load from local folder using pretrained_cfg_overlay
        local_model_path = "./models/dinov2_model/pytorch_model.bin"  # Adjust path as needed
        
        if os.path.exists(local_model_path):
            print(f"Loading model from local path: {local_model_path}")
            self.model = timm.create_model(
                "vit_small_patch14_dinov2", 
                pretrained=True,
                num_classes=0,
                pretrained_cfg_overlay=dict(file=local_model_path)
            )
        else:
            print("Local model not found, downloading from Hugging Face...")
            self.model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        
        self.model.to(self.device).eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        
        
    def load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded index with {self.index.ntotal} PDFs")
        else:
            self.index = faiss.IndexFlatL2(EMBED_DIM)
            self.metadata = {}
            print("Created new empty index")
    
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Create main paned window for resizable layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls and console
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel for results
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # Configure left frame
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(4, weight=1)  # Console output gets most space
        
        # Upload section
        upload_frame = ttk.LabelFrame(left_frame, text="Upload PDFs", padding="10")
        upload_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        upload_frame.columnconfigure(0, weight=1)
        
        ttk.Button(upload_frame, text="Select PDFs to Upload", 
                  command=self.select_pdfs_to_upload).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.process_btn = ttk.Button(upload_frame, text="Process Selected PDFs", 
                                     command=self.start_processing, state='disabled')
        self.process_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(upload_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Progress label
        self.progress_label = ttk.Label(upload_frame, text="Ready")
        self.progress_label.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # Search section
        search_frame = ttk.LabelFrame(left_frame, text="Search Similar PDFs", padding="10")
        search_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(0, weight=1)
        
        ttk.Button(search_frame, text="Select Query PDF", 
                  command=self.select_query_pdf).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Button(search_frame, text="Find Similar PDFs", 
                  command=self.find_similar_pdfs).grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Management section
        manage_frame = ttk.LabelFrame(left_frame, text="Database Management", padding="10")
        manage_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        manage_frame.columnconfigure(0, weight=1)
        
        ttk.Button(manage_frame, text="View All Records", 
                  command=self.view_all_records).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Button(manage_frame, text="Clear Database", 
                  command=self.clear_database).grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Status section
        status_frame = ttk.LabelFrame(left_frame, text="Status", padding="5")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack()
        
        # Console output section
        console_frame = ttk.LabelFrame(left_frame, text="Console Output", padding="10")
        console_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        
        # Console text widget with scrollbar
        self.console_text = scrolledtext.ScrolledText(console_frame, height=15, width=50, 
                                                     font=('Consolas', 9), bg='black', fg='white')
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Clear console button
        ttk.Button(console_frame, text="Clear Console", 
                  command=self.clear_console).grid(row=1, column=0, pady=(5, 0))
        
        # Results section (right panel)
        results_frame = ttk.LabelFrame(right_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create canvas with scrollbar for results
        self.canvas = Canvas(results_frame, bg='white')
        scrollbar = Scrollbar(results_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas, bg='white')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Initialize variables
        self.selected_pdfs = []
        self.query_pdf_path = None
        
    def setup_console_redirect(self):
        """Setup console output redirection"""
        self.console_redirect = ConsoleRedirect(self.console_text)
        # Redirect stdout to console widget
        sys.stdout = self.console_redirect
        
        # Print initial message
        print("PDF Similarity Search Application Started")
        print(f"Device: {self.device}")
        print(f"Model: {MODEL_NAME}")
        print(f"Images will be saved to: {IMAGES_DIR}")
        print("-" * 50)
        
    def clear_console(self):
        """Clear the console output"""
        self.console_text.delete(1.0, tk.END)
        
    def select_pdfs_to_upload(self):
        """Select multiple PDFs to upload"""
        filetypes = [("PDF files", "*.pdf"), ("All files", "*.*")]
        filenames = filedialog.askopenfilenames(
            title="Select PDFs to upload",
            filetypes=filetypes
        )
        
        if filenames:
            self.selected_pdfs = list(filenames)
            self.status_var.set(f"Selected {len(self.selected_pdfs)} PDFs for upload")
            self.process_btn.config(state='normal')
            print(f"Selected {len(self.selected_pdfs)} PDFs for processing:")
            for pdf in self.selected_pdfs:
                print(f"  - {os.path.basename(pdf)}")
    
    def start_processing(self):
        """Start PDF processing in a separate thread"""
        if not self.selected_pdfs:
            messagebox.showwarning("No PDFs Selected", "Please select PDFs to upload first.")
            return
        
        # Disable the process button during processing
        self.process_btn.config(state='disabled')
        
        # Reset progress bar
        self.progress_var.set(0)
        self.progress_label.config(text="Starting processing...")
        
        # Start processing in a separate thread to prevent GUI freezing
        threading.Thread(target=self.process_selected_pdfs, daemon=True).start()
    
    def process_selected_pdfs(self):
        """Process the selected PDFs with progress updates"""
        print("\n" + "="*50)
        print("STARTING PDF PROCESSING")
        print("="*50)
        
        stats = {'added': 0, 'skipped': 0, 'renamed': 0}
        total_pdfs = len(self.selected_pdfs)
        
        for i, pdf_path in enumerate(self.selected_pdfs):
            name = os.path.basename(pdf_path)
            
            # Update progress
            progress = ((i + 1) / total_pdfs) * 100
            self.progress_var.set(progress)
            self.progress_label.config(text=f"Processing {i+1}/{total_pdfs}: {name}")
            self.root.update_idletasks()
            
            try:
                print(f"\n[{i+1}/{total_pdfs}] Processing: {name}")
                
                result = self.process_pdf(pdf_path, name)
                if result == "added":
                    stats['added'] += 1
                elif result == "renamed":
                    stats['renamed'] += 1
                    stats['added'] += 1
                else:
                    stats['skipped'] += 1
                    
            except Exception as e:
                print(f"❌ Error processing {name}: {e}")
                stats['skipped'] += 1
        
        # Save index
        self.save_index()
        
        # Update UI
        self.selected_pdfs = []
        self.progress_var.set(100)
        self.progress_label.config(text="Processing complete!")
        self.status_var.set(f"Complete: {stats['added']} added ({stats['renamed']} renamed), {stats['skipped']} skipped")
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print(f"Added: {stats['added']}, Renamed: {stats['renamed']}, Skipped: {stats['skipped']}")
        print("="*50)
        
        # Re-enable the process button after a delay
        self.root.after(2000, lambda: self.process_btn.config(state='normal'))
        
    def select_query_pdf(self):
        """Select a PDF for similarity search"""
        filetypes = [("PDF files", "*.pdf"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(
            title="Select query PDF",
            filetypes=filetypes
        )
        
        if filename:
            self.query_pdf_path = filename
            self.status_var.set(f"Query PDF selected: {os.path.basename(filename)}")
            print(f"\nQuery PDF selected: {os.path.basename(filename)}")
    
    def find_similar_pdfs(self):
        """Find and display similar PDFs, adding query PDF to index if not duplicate"""
        if not self.query_pdf_path:
            messagebox.showwarning("No Query PDF", "Please select a query PDF first.")
            return

        print("\n" + "-"*50)
        print("SEARCHING FOR SIMILAR PDFs")
        print("-"*50)

        self.status_var.set("Processing query PDF...")
        self.root.update()

        # Check if query PDF should be added to index
        query_name = os.path.basename(self.query_pdf_path)
        
        # First, check if query PDF is already in the index
        query_added = False
        final_query_name = query_name
        
        if self.index.ntotal > 0:
            # Check for duplicates before adding
            query_hash = self.sha256_hash(self.query_pdf_path)
            query_img = self.pdf_to_image(self.query_pdf_path)
            
            if query_img is not None:
                query_emb = self.extract_embedding(query_img)
                dup_type, final_name = self.check_duplicate_enhanced(query_name, query_hash, query_emb)
                
                if not dup_type:
                    # Add to index (this will save the image)
                    result = self.process_pdf(self.query_pdf_path, final_name)
                    if result in ["added", "renamed"]:
                        query_added = True
                        final_query_name = final_name
                        print(f"✅ Added query PDF to index: {final_name}")
                        if result == "renamed":
                            print(f"  📝 Renamed from: {query_name}")
                        self.save_index()
                    else:
                        print(f"❌ Failed to add query PDF to index: {query_name}")
                else:
                    print(f"⚠️ Query PDF already exists in index (duplicate by {dup_type}): {query_name}")
            else:
                print(f"❌ Could not process query PDF: {query_name}")
        else:
            # Empty index, definitely add the query PDF
            query_img = self.pdf_to_image(self.query_pdf_path)
            
            if query_img is not None:
                result = self.process_pdf(self.query_pdf_path, query_name)
                if result in ["added", "renamed"]:
                    query_added = True
                    print(f"✅ Added query PDF to index (first PDF): {query_name}")
                    self.save_index()

        # Now proceed with similarity search
        if self.index.ntotal == 0:
            messagebox.showwarning("Empty Index", "No PDFs in the database after processing.")
            return

        self.status_var.set("Searching for similar PDFs...")
        self.root.update()

        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Display query PDF
        query_frame = Frame(self.scrollable_frame, bg='white', relief=tk.RIDGE, bd=2)
        query_frame.pack(fill=tk.X, padx=10, pady=10)
        
        query_label_text = "Query PDF:"
        if query_added:
            query_label_text += " (Added to Index)"
        
        Label(query_frame, text=query_label_text, font=('Arial', 14, 'bold'), bg='white').pack()
        Label(query_frame, text=final_query_name, bg='white').pack()

        # Display query PDF image - load from saved image if available
        query_image = self.get_saved_image(final_query_name)
        if query_image is None:
            query_image = self.pdf_to_image(self.query_pdf_path)
        
        if query_image:
            self.display_pdf_image(query_frame, query_image)

        # Find similar PDFs (excluding the query PDF itself if it was just added)
        k_search = 10 if query_added else 9  # Get one extra if query was added
        similar_pdfs = self.find_similar(self.query_pdf_path, k=k_search)

        # Filter out the query PDF from results if it was just added
        if query_added:
            similar_pdfs = [(name, dist, pid) for name, dist, pid in similar_pdfs if name != final_query_name]
            similar_pdfs = similar_pdfs[:9]  # Keep only top 9 results

        if not similar_pdfs:
            Label(self.scrollable_frame, text="No other similar PDFs found.",
                font=('Arial', 12), bg='white').pack(pady=20)
            print("No other similar PDFs found.")
        else:
            # Display results
            Label(self.scrollable_frame, text=f"Found {len(similar_pdfs)} similar PDFs:",
                font=('Arial', 14, 'bold'), bg='white').pack(pady=10)
            print(f"Found {len(similar_pdfs)} similar PDFs:")

            for i, (pdf_name, dist, pid) in enumerate(similar_pdfs, 1):
                print(f" {i}. {pdf_name} (distance: {dist:.4f})")

            # Create grid for results
            results_container = Frame(self.scrollable_frame, bg='white')
            results_container.pack(fill=tk.BOTH, expand=True, padx=10)

            for i, (pdf_name, dist, pid) in enumerate(similar_pdfs):
                row = i // 3
                col = i % 3

                # Create frame for each result
                result_frame = Frame(results_container, bg='white', relief=tk.RIDGE, bd=1)
                result_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

                # PDF name and distance
                Label(result_frame, text=f"{i+1}. {pdf_name}",
                    font=('Arial', 10, 'bold'), bg='white', wraplength=250).pack()
                Label(result_frame, text=f"Distance: {dist:.4f}",
                    font=('Arial', 9), bg='white').pack()

                # Display PDF image from saved image
                saved_img = self.get_saved_image(pdf_name)
                if saved_img:
                    self.display_pdf_image(result_frame, saved_img)

                # Add delete button
                delete_btn = Button(result_frame, text="Delete", bg='red', fg='white',
                                  command=lambda p=pid, n=pdf_name: self.delete_record(p, n))
                delete_btn.pack(pady=2)

            # Configure grid weights
            for col in range(3):
                results_container.columnconfigure(col, weight=1)

        self.status_var.set("Search completed")
        print("Search completed.")

    def view_all_records(self):
        """Display all records in the database"""
        if self.index.ntotal == 0:
            messagebox.showinfo("Empty Database", "No records in the database.")
            return

        print(f"\n📋 Viewing all {self.index.ntotal} records in database:")

        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Title
        Label(self.scrollable_frame, text=f"All Records ({self.index.ntotal} total)",
              font=('Arial', 16, 'bold'), bg='white').pack(pady=10)

        # Create grid for all records
        results_container = Frame(self.scrollable_frame, bg='white')
        results_container.pack(fill=tk.BOTH, expand=True, padx=10)

        # Sort records by name for consistent display
        sorted_records = sorted(self.metadata.items(), key=lambda x: x[1]['name'])

        for i, (pid, record) in enumerate(sorted_records):
            row = i // 3
            col = i % 3

            # Create frame for each record
            record_frame = Frame(results_container, bg='white', relief=tk.RIDGE, bd=1)
            record_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            # Record info
            Label(record_frame, text=f"{i+1}. {record['name']}",
                  font=('Arial', 10, 'bold'), bg='white', wraplength=250).pack()
            
            # Add timestamp if available
            if 'time' in record:
                time_str = record['time'][:19]  # Remove milliseconds
                Label(record_frame, text=f"Added: {time_str}",
                      font=('Arial', 8), bg='white').pack()

            # Display image
            saved_img = self.get_saved_image(record['name'])
            if saved_img:
                self.display_pdf_image(record_frame, saved_img)

            # Add delete button
            delete_btn = Button(record_frame, text="Delete", bg='red', fg='white',
                              command=lambda p=pid, n=record['name']: self.delete_record(p, n))
            delete_btn.pack(pady=2)

            print(f" {i+1}. {record['name']} (ID: {pid})")

        # Configure grid weights
        for col in range(3):
            results_container.columnconfigure(col, weight=1)

        self.status_var.set(f"Viewing all {self.index.ntotal} records")

    def delete_record(self, record_id, record_name):
        """Delete a specific record from the database"""
        result = messagebox.askyesno("Confirm Deletion", 
                                   f"Are you sure you want to delete '{record_name}'?")
        if not result:
            return

        try:
            print(f"\n🗑️ Deleting record: {record_name} (ID: {record_id})")
            
            # Remove image file if it exists
            if record_id in self.metadata and 'image_path' in self.metadata[record_id]:
                image_path = self.metadata[record_id]['image_path']
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"  🗑️ Deleted image file: {os.path.basename(image_path)}")

            # Since FAISS doesn't support direct deletion, we need to rebuild the index
            self.rebuild_index_without_record(record_id)
            
            print(f"  ✅ Successfully deleted: {record_name}")
            self.status_var.set(f"Deleted: {record_name}")
            
            # Simply remove the deleted record's widget from the current display
            # Find and remove the specific record widget
            for widget in self.scrollable_frame.winfo_children():
                if hasattr(widget, 'winfo_children'):
                    for child in widget.winfo_children():
                        if isinstance(child, Label) and record_name in child.cget('text'):
                            widget.destroy()
                            break
            
        except Exception as e:
            print(f"  ❌ Error deleting record: {e}")
            messagebox.showerror("Deletion Error", f"Failed to delete record: {e}")

    def rebuild_index_without_record(self, record_to_delete):
        """Rebuild the FAISS index without a specific record"""
        # Get all embeddings except the one to delete
        remaining_records = {pid: meta for pid, meta in self.metadata.items() if pid != record_to_delete}
        
        if not remaining_records:
            # If no records left, create empty index
            self.index = faiss.IndexFlatL2(EMBED_DIM)
            self.metadata = {}
            self.save_index()
            return

        # Create new index
        new_index = faiss.IndexFlatL2(EMBED_DIM)
        new_metadata = {}
        
        # Re-add all remaining records
        for old_pid, meta in remaining_records.items():
            # Load the image and extract embedding
            saved_img = self.get_saved_image(meta['name'])
            if saved_img:
                emb = self.extract_embedding(saved_img)
                new_index.add(emb.reshape(1, -1))
                
                # Assign new ID
                new_pid = str(new_index.ntotal - 1)
                new_metadata[new_pid] = meta
        
        # Replace old index and metadata
        self.index = new_index
        self.metadata = new_metadata
        self.save_index()
        
        print(f"  🔄 Rebuilt index with {self.index.ntotal} records")

    def clear_database(self):
        """Clear all records from the database"""
        if self.index.ntotal == 0:
            messagebox.showinfo("Empty Database", "Database is already empty.")
            return

        result = messagebox.askyesno("Confirm Clear Database", 
                                   f"Are you sure you want to delete all {self.index.ntotal} records?")
        if not result:
            return

        try:
            print(f"\n🗑️ Clearing database with {self.index.ntotal} records...")
            
            # Delete all image files
            for pid, meta in self.metadata.items():
                if 'image_path' in meta and os.path.exists(meta['image_path']):
                    os.remove(meta['image_path'])
            
            # Clear images directory of any remaining files
            for img_file in Path(IMAGES_DIR).glob("*.png"):
                img_file.unlink()
            
            # Create new empty index
            self.index = faiss.IndexFlatL2(EMBED_DIM)
            self.metadata = {}
            self.save_index()
            
            # Clear results display
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            print("  ✅ Database cleared successfully")
            self.status_var.set("Database cleared")
            
        except Exception as e:
            print(f"  ❌ Error clearing database: {e}")
            messagebox.showerror("Clear Error", f"Failed to clear database: {e}")

    def get_saved_image(self, pdf_name):
        """Load saved image for a PDF"""
        # Create image filename from PDF name
        image_name = os.path.splitext(pdf_name)[0] + ".png"
        image_path = Path(IMAGES_DIR) / image_name
        
        if image_path.exists():
            try:
                return Image.open(image_path)
            except Exception as e:
                print(f"❌ Error loading saved image {image_name}: {e}")
                return None
        return None

    def save_pdf_image(self, pil_image, pdf_name):
        """Save PDF image to images directory"""
        try:
            # Create image filename from PDF name
            image_name = os.path.splitext(pdf_name)[0] + ".png"
            image_path = Path(IMAGES_DIR) / image_name
            
            # Resize image to standard size for storage
            img_copy = pil_image.copy()
            img_copy.thumbnail((SAVED_IMG_WIDTH, SAVED_IMG_HEIGHT))
            
            # Save as PNG
            img_copy.save(image_path, "PNG", optimize=True)
            print(f"  💾 Saved image: {image_name}")
            return str(image_path)
        except Exception as e:
            print(f"  ❌ Error saving image for {pdf_name}: {e}")
            return None
        
    def display_pdf_image(self, parent, pil_image):
        """Display a PIL image in a tkinter widget"""
        # Resize image to fit
        img_copy = pil_image.copy()
        img_copy.thumbnail((IMG_MAX_WIDTH, IMG_MAX_HEIGHT))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img_copy)
        
        # Create label and keep reference
        img_label = Label(parent, image=photo, bg='white')
        img_label.image = photo  # Keep a reference
        img_label.pack(pady=5)
        
    def generate_unique_name(self, base_name):
        """Generate a unique name by appending a suffix if needed"""
        existing_names = {meta['name'] for meta in self.metadata.values()}
        
        if base_name not in existing_names:
            return base_name
        
        # Extract name and extension
        name_part, ext = os.path.splitext(base_name)
        
        # Try numbered suffixes
        counter = 1
        while True:
            new_name = f"{name_part}_{counter:02d}{ext}"
            if new_name not in existing_names:
                return new_name
            counter += 1
            
            # Safety check to prevent infinite loop
            if counter > 999:
                import uuid
                unique_id = str(uuid.uuid4())[:8]
                return f"{name_part}_{unique_id}{ext}"
    
    def check_duplicate_enhanced(self, name, file_hash, emb):
        """Enhanced duplicate checking with unique name generation"""
        # Hash check - this is always a true duplicate
        for mid, m in self.metadata.items():
            if m.get('hash') and m['hash'] == file_hash:
                return "hash", name
        
        # Embedding check - this is always a true duplicate (same content)
        if self.index.ntotal > 0:
            dists, idxs = self.index.search(emb.reshape(1, -1), 1)
            if dists[0][0] < SIMILARITY_THRESHOLD:
                return "content", name
        
        # Name check - only return duplicate if name exists, but generate unique name
        existing_names = {meta['name'] for meta in self.metadata.values()}
        if name in existing_names:
            unique_name = self.generate_unique_name(name)
            return None, unique_name  # Not a duplicate, but name was changed
        
        return None, name
    
    # Helper methods (adapted from original code)
    def save_index(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"💾 Index saved ({self.index.ntotal} entries)")
    
    def sha256_hash(self, path):
        """Calculate SHA256 hash of file"""
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                h.update(chunk)
        return h.hexdigest()
    
    def pdf_to_image(self, path):
        """Convert first PDF page to PIL Image"""
        try:
            doc = fitz.open(path)
            if doc.page_count == 0:
                doc.close()
                return None
            
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            doc.close()
            return img
        except:
            try:
                doc = fitz.open(path)
                clean_path = path.replace('.pdf', '_cleaned.pdf')
                doc.save(clean_path, garbage=3, clean=True)
                doc.close()
                img = self.pdf_to_image(clean_path)
                os.remove(clean_path)
                return img
            except:
                return None
    
    def extract_embedding(self, img):
        """Extract embedding from image using DINOv2"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(t)
            emb = torch.nn.functional.normalize(feat, dim=-1)
        
        return emb.cpu().numpy().astype('float32')[0]
    
    def is_duplicate(self, name, file_hash, emb):
        """Check if PDF is duplicate (legacy method, kept for compatibility)"""
        # Name check
        if any(m.get('name') == name for m in self.metadata.values()):
            return "name"
        
        # Hash check
        for mid, m in self.metadata.items():
            if m.get('hash') and m['hash'] == file_hash:
                return "hash"
        
        # Embedding check
        if self.index.ntotal > 0:
            dists, idxs = self.index.search(emb.reshape(1, -1), 1)
            if dists[0][0] < SIMILARITY_THRESHOLD:
                return "content"
        
        return None
    
    def process_pdf(self, path, name):
        """Process a single PDF with enhanced duplicate handling"""
        h = self.sha256_hash(path)
        img = self.pdf_to_image(path)
        
        if img is None:
            print("  ❌ Could not render PDF")
            return "failed"
        
        emb = self.extract_embedding(img)
        dup_type, final_name = self.check_duplicate_enhanced(name, h, emb)
        
        if dup_type:
            print(f"  ⚠️ Duplicate by {dup_type}")
            return "skipped"
        
        # Determine if name was changed
        was_renamed = final_name != name
        
        # Save the image with the final name
        saved_image_path = self.save_pdf_image(img, final_name)
        if saved_image_path is None:
            print("  ❌ Could not save image")
            return "failed"
        
        # Add to index & metadata
        self.index.add(emb.reshape(1, -1))
        pid = str(self.index.ntotal - 1)
        self.metadata[pid] = {
            'name': final_name,
            'original_name': name if was_renamed else final_name,
            'image_path': saved_image_path,
            'hash': h,
            'time': datetime.now().isoformat()
        }
        
        if was_renamed:
            print(f"  📝 Renamed to: {final_name}")
            print("  ✅ Added successfully (renamed)")
            return "renamed"
        else:
            print("  ✅ Added successfully")
            return "added"
    
    def find_similar(self, path, k=5):
        """Find similar PDFs"""
        img = self.pdf_to_image(path)
        if img is None:
            print("❌ Cannot render query PDF")
            return []
        
        emb = self.extract_embedding(img).reshape(1, -1)
        k = min(k, self.index.ntotal)
        dists, idxs = self.index.search(emb, k)
        
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            m = self.metadata[str(idx)]
            results.append((m['name'], float(dist), str(idx)))
        
        return results

def main():
    root = tk.Tk()
    app = PDFSimilarityApp(root)
    
    # Handle window closing to restore stdout
    def on_closing():
        sys.stdout = sys.__stdout__  # Restore original stdout
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()