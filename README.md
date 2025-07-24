# Similar Drawing - PDF Similarity Search Tool
Desktop application for finding visually similar PDF documents using AI image embeddings. Built with Python, this tool helps you organize and search through large collections of PDF files based on visual content similarity.


- **Visual PDF Search**: Find similar PDFs based on visual content using DINOv2 embeddings
- **Batch Processing**: Upload and process multiple PDFs simultaneously
- **Progress Tracking**: Monitor processing status with progress bars and console output
- **Interactive GUI**: User-friendly interface with scrollable results and image previews

### **Duplicate Handling**
The application features a 3-tier duplicate detection system:

1. **Hash-based Detection**: Identifies identical files using SHA256 checksums
2. **Content-based Detection**: Detects visually identical content using embedding similarity (threshold: 1e-7)
3. **Name-based Handling**: Automatically renames files with duplicate names using incremental suffixes

**Duplicate Treatment Options**:
- **True Duplicates** (hash/content): Automatically skipped with clear notification
- **Name Conflicts**: Files are automatically renamed (e.g., `document.pdf` → `document_01.pdf`)
- **Query PDF Integration**: When searching, the query PDF is automatically added to the database if it's not a duplicate

### **Database Management**
- **View All Records**: Browse all indexed PDFs in a grid layout with thumbnails
- **Individual Deletion**: Remove specific records with confirmation dialogs
- **Bulk Operations**: Clear entire database when needed
- **Persistent Storage**: FAISS index with JSON metadata for fast retrieval

### **Visual Features**
- **PDF Thumbnails**: Automatic generation and storage of PDF preview images
- **Grid Layout**: Organized display of results in 3-column format
- **Image Caching**: Efficient storage and retrieval of PDF thumbnails
- **Responsive UI**: Resizable panels and scrollable content areas


# Motivation
With years of experience in metalworking and roles as a designer, estimator, and buyer, I understand the daily challenges of manufacturing. A job well done often comes from learning from past project experiences. This project aims to make it easier to find old finished jobs that can help us accomplish our current work.
* **For Mechanical Designers**: Upload a sketch or drawing and find similar parts from your archive. See how others solved similar problems and what approaches worked best.
* **For Production Preparers**: Discover similar components and learn from proven manufacturing sequences, tooling setups, and process parameters that delivered results.
* **For Estimators and Budgeters**: Improve pricing accuracy by referencing historical data. Find similar quoted parts, review actual costs, lead times, and margins to create more competitive and realistic estimates.
* **For Buyers**: Rely on your supply chain history. Identify similar components and see which suppliers delivered quality on time and at what price—avoid past mistakes and strengthen supplier negotiations.
* **For Production Analysts**: Understand how similar parts performed in production—cycle times, quality issues, rework rates—so you can make informed, data-driven decisions for new components.
I'm not a programmer, but someone who learned Python and makes use of LLMs to develop some tools that help me at work, so please take that into account.



## Installation

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
pip install timm
pip install PyMuPDF
pip install Pillow
pip install numpy
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/bmg92silva/Similar-Drawings.git
cd similar-drawing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python Similar_Drawing.py
```

## Usage

### **Getting Started**
1. **Launch the Application**: Run `python Similar_Drawing.py`
2. **Upload PDFs**: Click "Select PDFs to Upload" and choose your PDF files
3. **Process Files**: Click "Process Selected PDFs" to add them to the database
4. **Search**: Select a query PDF and click "Find Similar PDFs"

### **Search Workflow**
1. Select a query PDF using "Select Query PDF"
2. Click "Find Similar PDFs"
3. The system will:
   - Check if the query PDF is already in the database
   - Add it automatically if it's not a duplicate
   - Display the top 9 most similar PDFs with distance scores
   - Show thumbnails for visual comparison

### **Database Management**
- **View All Records**: See all indexed PDFs with metadata and thumbnails
- **Delete Individual Records**: Remove specific PDFs from the index
- **Clear Database**: Reset the entire database (with confirmation)

## Technical Details

### **AI Model**
- **Architecture**: Vision Transformer (ViT) with DINOv2
- **Model**: `timm/vit_small_patch14_dinov2.lvd142m`
- **Embedding Dimension**: 384
- **Device Support**: Automatic CPU/GPU detection

### **Storage**
- **Index**: FAISS FlatL2 for similarity search
- **Metadata**: JSON storage for PDF information
- **Images**: PNG thumbnails (300x200px) for quick preview
- **File Structure**:
  ```
  ├── pdf_embeddings.index    # FAISS index file
  ├── pdf_metadata.json       # PDF metadata
  └── pdf_images/             # Thumbnail storage
  ```


## Duplicate Detection System

| Detection Type | Method | Action | Example |
|----------------|--------|--------|---------|
| **Hash Duplicate** | SHA256 checksum | Skip with notification | Identical file content |
| **Content Duplicate** | Embedding similarity < 1e-7 | Skip with notification | Same visual content, different file |
| **Name Conflict** | Filename comparison | Auto-rename with suffix | `doc.pdf` → `doc_01.pdf` |
| **Unique File** | No matches found | Add to database | Standard processing |



## Known Issues & Limitations

- **PDF Compatibility**: Some encrypted or corrupted PDFs may not process correctly
- **Memory Usage**: Large batch processing may take long time
- **Model Download**: First run requires downloading the DINOv2 model (~400MB)

## Future Enhancements

- [ ] Multi-page PDF analysis
- [ ] Custom similarity thresholds
- [ ] Web interface option
- [ ] OCR integration for text-based similarity

**⭐ Star this repository if you find it useful!**
For questions, issues, or feature requests, please open an issue on GitHub.
