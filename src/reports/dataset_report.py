"""
Dataset Report Generator
Creates professional PDF reports with comprehensive dataset analysis
"""

import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (11, 8.5)  # Letter size
plt.rcParams['font.size'] = 10


class DatasetReportGenerator:
    """Generate comprehensive PDF reports for datasets"""
    
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.train_folder = os.path.join(dataset_folder, 'train')
        self.debug_folder = os.path.join(dataset_folder, 'debug')
        self.stats = {}
        self.images_info = {'train': [], 'debug': []}
        self.avg_train_img = None # To store cumulative average for heatmap
        
    def analyze_dataset(self):
        """Analyze the entire dataset and collect statistics"""
        print("Analyzing dataset...")
        
        # Analyze train folder
        if os.path.exists(self.train_folder):
            self.images_info['train'] = self._analyze_folder(self.train_folder)
        
        # Analyze debug folder
        if os.path.exists(self.debug_folder):
            self.images_info['debug'] = self._analyze_folder(self.debug_folder)
        
        # Calculate overall statistics
        self._calculate_statistics()
        
        # Compute advanced analytics (Heatmap & Background Noise)
        self._compute_train_heatmap_analytics()
        
        print(f"Analysis complete: {self.stats['total_images']} images analyzed")
        
    def _analyze_folder(self, folder_path):
        """Analyze all images in a folder"""
        images_info = []
        
        if not os.path.exists(folder_path):
            return images_info
        
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(folder_path, filename)
                
                # Load image
                img = cv2.imread(filepath)
                if img is None:
                    continue
                
                # Calculate image metrics
                centroid, diameter = self._calculate_centroid_and_diameter(img)
                
                info = {
                    'filename': filename,
                    'path': filepath,
                    'size': os.path.getsize(filepath),
                    'shape': img.shape,
                    'mean_brightness': np.mean(img),
                    'std_brightness': np.std(img),
                    'sharpness': self._calculate_sharpness(img),
                    'contrast': self._calculate_contrast(img),
                    'centroid': centroid,  # (x, y) position
                    'diameter': diameter,  # Object diameter in pixels
                }
                
                images_info.append(info)
        
        return images_info
    
    def _calculate_centroid_and_diameter(self, img):
        """Calculate object centroid and diameter for alignment analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Threshold to find object (assuming dark object on light background or vice versa)
            # Try both directions and use the one with larger area
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours for both
            contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Use the largest contour
            all_contours = list(contours1) + list(contours2)
            if not all_contours:
                return (img.shape[1]/2, img.shape[0]/2), 0  # Image center if no object found
            
            largest_contour = max(all_contours, key=cv2.contourArea)
            
            # Calculate moments
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return (img.shape[1]/2, img.shape[0]/2), 0
            
            # Centroid
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            # Diameter (approximate using bounding circle)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            diameter = radius * 2
            
            return (cx, cy), diameter
            
        except Exception as e:
            print(f"Centroid calculation failed: {e}")
            return (img.shape[1]/2, img.shape[0]/2), 0
    
    def _calculate_sharpness(self, img):
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _calculate_contrast(self, img):
        """Calculate image contrast (std deviation)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.std(gray)
    
    def _calculate_statistics(self):
        """Calculate overall dataset statistics"""
        train_count = len(self.images_info['train'])
        debug_count = len(self.images_info['debug'])
        total_count = train_count + debug_count
        
        self.stats = {
            'total_images': total_count,
            'train_images': train_count,
            'debug_images': debug_count,
            'success_rate': (train_count / total_count * 100) if total_count > 0 else 0,
            'dataset_name': os.path.basename(self.dataset_folder),
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Calculate average metrics for train images
        if train_count > 0:
            self.stats['train_avg_sharpness'] = np.mean([img['sharpness'] for img in self.images_info['train']])
            self.stats['train_avg_brightness'] = np.mean([img['mean_brightness'] for img in self.images_info['train']])
            self.stats['train_avg_contrast'] = np.mean([img['contrast'] for img in self.images_info['train']])
        else:
            self.stats['train_avg_sharpness'] = 0
            self.stats['train_avg_brightness'] = 0
            self.stats['train_avg_contrast'] = 0
        
        # Calculate average metrics for debug images
        if debug_count > 0:
            self.stats['debug_avg_sharpness'] = np.mean([img['sharpness'] for img in self.images_info['debug']])
            self.stats['debug_avg_brightness'] = np.mean([img['mean_brightness'] for img in self.images_info['debug']])
            self.stats['debug_avg_contrast'] = np.mean([img['contrast'] for img in self.images_info['debug']])
        else:
            self.stats['debug_avg_sharpness'] = 0
            self.stats['debug_avg_brightness'] = 0
            self.stats['debug_avg_contrast'] = 0
        
        # Calculate alignment metrics (centroid variance and scale stability)
        # Critical for PatchCore which is sensitive to object position
        if train_count > 1:
            centroids = [img['centroid'] for img in self.images_info['train']]
            diameters = [img['diameter'] for img in self.images_info['train'] if img['diameter'] > 0]
            
            # Centroid variance (how much the object moves)
            centroids_x = [c[0] for c in centroids]
            centroids_y = [c[1] for c in centroids]
            self.stats['centroid_std_x'] = np.std(centroids_x)
            self.stats['centroid_std_y'] = np.std(centroids_y)
            self.stats['centroid_variance'] = np.sqrt(self.stats['centroid_std_x']**2 + self.stats['centroid_std_y']**2)
            
            # Scale stability (diameter variation)
            if diameters:
                self.stats['diameter_mean'] = np.mean(diameters)
                self.stats['diameter_std'] = np.std(diameters)
                self.stats['diameter_cv'] = (self.stats['diameter_std'] / self.stats['diameter_mean']) * 100  # Coefficient of variation %
            else:
                self.stats['diameter_mean'] = 0
                self.stats['diameter_std'] = 0
                self.stats['diameter_cv'] = 0
        else:
            self.stats['centroid_std_x'] = 0
            self.stats['centroid_std_y'] = 0
            self.stats['centroid_variance'] = 0
            self.stats['diameter_mean'] = 0
            self.stats['diameter_std'] = 0
            self.stats['diameter_cv'] = 0
            
        # Detect outliers
        self.outliers = self._detect_outliers()
        self.stats['outlier_count'] = len(self.outliers)
            
    def _detect_outliers(self):
        """Detect outliers in the training set using Z-score analysis"""
        outliers = []
        if not self.images_info['train']:
            return outliers
            
        train_imgs = self.images_info['train']
        
        # Metrics to analyze for outliers
        metrics = {
            'sharpness': 'Sharpness',
            'mean_brightness': 'Brightness', 
            'contrast': 'Contrast'
        }
        
        # Calculate stats for z-score
        stats = {}
        for key in metrics:
            values = [img[key] for img in train_imgs]
            stats[key] = {'mean': np.mean(values), 'std': np.std(values)}
        
        # Minimum STD thresholds to prevent "over-cleaning"
        # If the dataset is already very consistent (low std), don't flag small deviations as outliers
        min_std_thresholds = {
            'sharpness': 5.0,        # Don't flag if sharpess varies by less than 5
            'mean_brightness': 3.0,  # Don't flag if brightness varies by less than 3 (out of 255)
            'contrast': 3.0          # Don't flag if contrast varies by less than 3
        }

        # Identify outliers (Z-score > 2.0)
        for img in train_imgs:
            reasons = []
            is_outlier = False
            
            for key, label in metrics.items():
                if stats[key]['std'] < min_std_thresholds.get(key, 1.0):
                    continue  # Skip if variation is already very low (stable)
                    
                val = img[key]
                z_score = abs(val - stats[key]['mean']) / stats[key]['std']
                
                # SENSITIVE THRESHOLD: Z > 2.0 (approx 5% of data)
                if z_score > 2.0:
                    is_outlier = True
                    direction = "LOW" if val < stats[key]['mean'] else "HIGH"
                    reasons.append(f"{label} {direction} (Z={z_score:.1f}, Val={val:.1f})")
            
            if is_outlier:
                img_copy = img.copy()
                img_copy['outlier_reasons'] = reasons
                outliers.append(img_copy)
                
        # Sort by number of reasons then filename
        outliers.sort(key=lambda x: (len(x['outlier_reasons']), x['filename']), reverse=True)
        return outliers
    
    def move_outliers_to_debug(self):
        """Move detected outliers from train to debug folder"""
        if not self.outliers:
            return 0
            
        moved_count = 0
        print(f"Moving {len(self.outliers)} outliers to debug folder...")
        
        for img in self.outliers:
            try:
                src = img['path']
                dst = os.path.join(self.debug_folder, img['filename'])
                
                # Move file
                os.rename(src, dst)
                moved_count += 1
                
                # Update internal lists (simple removal/addition)
                # Note: to fully refresh stats, re-running analyze_dataset is best
                
            except Exception as e:
                print(f"Failed to move {img['filename']}: {e}")
                
        return moved_count

    def _compute_train_heatmap_analytics(self):
        """Compute average image heatmap and background noise stats"""
        train_imgs = self.images_info['train']
        if not train_imgs:
            return
            
        print("Computing heatmap and background analytics...")
        
        # Initialize accumulator
        first_img = cv2.imread(train_imgs[0]['path'])
        if first_img is None:
            return
            
        h, w = first_img.shape[:2]
        accumulator = np.zeros((h, w, 3), dtype=np.float64)
        count = 0
        
        # Accumulate all images
        for info in train_imgs:
            img = cv2.imread(info['path'])
            if img is not None and img.shape == first_img.shape:
                accumulator += img
                count += 1
                
        if count > 0:
            # Compute average
            self.avg_train_img = (accumulator / count).astype(np.uint8)
            
            # Analyze Background Noise (Corners)
            # Define corners (assuming centered object)
            # 10% size of image from each corner
            cw, ch = int(w*0.1), int(h*0.1)
            
            # Extract corners from average image
            tl = self.avg_train_img[0:ch, 0:cw]
            tr = self.avg_train_img[0:ch, w-cw:w]
            bl = self.avg_train_img[h-ch:h, 0:cw]
            br = self.avg_train_img[h-ch:h, w-cw:w]
            
            corners = np.concatenate([tl.flatten(), tr.flatten(), bl.flatten(), br.flatten()])
            
            self.stats['bg_noise_mean'] = np.mean(corners)
            self.stats['bg_noise_std'] = np.std(corners)
            self.stats['bg_noise_max'] = np.max(corners)

    def _calculate_quality_score(self):
        """Calculate overall dataset quality score (0-100)"""
        score = 100
        penalties = []
        
        # 1. Outlier Penalty (High impact)
        outlier_count = self.stats.get('outlier_count', 0)
        if outlier_count > 0:
            p = min(40, outlier_count * 5)
            score -= p
            penalties.append(f"-{p} pts: {outlier_count} detected outliers (Critical)")
        
        # 2. Alignment Penalty (Medium impact)
        variance = self.stats.get('centroid_variance', 0)
        if variance > 2.0:
            p = min(30, int((variance - 2.0) * 5))
            score -= p
            penalties.append(f"-{p} pts: Unstable alignment ({variance:.2f}px variance)")
            
        # 3. Background Penalty (Low impact but important)
        bg_noise = self.stats.get('bg_noise_mean', 0)
        if bg_noise > 2.0:
            p = min(20, int((bg_noise - 2.0) * 2))
            score -= p
            penalties.append(f"-{p} pts: Background noise detected ({bg_noise:.1f}/255)")
            
        # 4. Quantity Penalty
        train_count = self.stats.get('train_images', 0)
        if train_count < 20:
            p = 20
            score -= p
            penalties.append(f"-{p} pts: Dataset too small (<20 images)")
            
        return max(0, score), penalties

    def generate_report(self, output_path):
        """Generate a comprehensive PDF report"""
        print(f"Generating report: {output_path}")
        
        with PdfPages(output_path) as pdf:
            # Page 1: Title and Summary
            self._create_title_page(pdf)
            
            # Page 2: Dataset Overview Statistics
            self._create_overview_page(pdf)
            
            # Page 3: Quality Metrics Distributions
            self._create_metrics_page(pdf)
            
            # Page 4: Alignment Verification (PatchCore Critical)
            self._create_alignment_page(pdf)
            
            # Page 5: Outlier Analysis (Training Set Purity)
            self._create_outlier_page(pdf)
            
            # Page 6: Advanced Stability (Heatmap & Noise)
            self._create_advanced_page(pdf)
            
            # Page 7: Image Quality Analysis
            self._create_quality_analysis_page(pdf)
            
            # Page 8: Sample Images
            self._create_samples_page(pdf)
            
            # Page 8: Sample Images
            self._create_samples_page(pdf)
            
            # Page 9: Conclusion & Recommendations
            self._create_conclusion_page(pdf)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = f'Dataset Report - {self.stats["dataset_name"]}'
            d['Author'] = 'InspectionVisionCamera System'
            d['Subject'] = 'Dataset Quality Analysis'
            d['Keywords'] = 'Dataset, Quality Analysis, Image Processing'
            d['CreationDate'] = datetime.now()
        
        print(f"✓ Report generated: {output_path}")
    
    def _create_outlier_page(self, pdf):
        """Create outlier analysis page for training set purity"""
        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Training Set Outlier Analysis (Memory Bank Purity)', fontsize=16, fontweight='bold')
        
        train_imgs = self.images_info['train']
        
        if not train_imgs:
            plt.text(0.5, 0.5, "No training images to analyze", ha='center', fontsize=12)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            return

        # 1. Sharpness Outlier Plot
        ax1 = fig.add_subplot(gs[0, 0])
        sharpness = [img['sharpness'] for img in train_imgs]
        mean_sharp = np.mean(sharpness)
        std_sharp = np.std(sharpness)
        
        # Plot all points
        ax1.scatter(range(len(sharpness)), sharpness, alpha=0.5, label='Normal', c='#4CAF50')
        
        # Highlight outliers
        outlier_indices = []
        outlier_values = []
        for i, val in enumerate(sharpness):
            z_score = abs(val - mean_sharp) / std_sharp if std_sharp > 0 else 0
            if z_score > 2.0:
                outlier_indices.append(i)
                outlier_values.append(val)
        
        if outlier_indices:
            ax1.scatter(outlier_indices, outlier_values, c='red', s=50, label='Outlier (Z>2.0)')
            # Annotate specific point mentioned by user if present (around 1400)
            for i, val in zip(outlier_indices, outlier_values):
                ax1.annotate(f"{val:.0f}", (i, val), xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax1.axhline(mean_sharp, color='blue', linestyle='--', label=f'Mean: {mean_sharp:.1f}')
        ax1.axhline(mean_sharp + 2.0*std_sharp, color='orange', linestyle=':', label='±2.0σ Threshold')
        ax1.axhline(mean_sharp - 2.0*std_sharp, color='orange', linestyle=':')
        
        ax1.set_title('Sharpness Consistency')
        ax1.set_xlabel('Image Index')
        ax1.set_ylabel('Sharpness (Laplacian Var)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Brightness vs Contrast Outlier Plot
        ax2 = fig.add_subplot(gs[0, 1])
        bright = [img['mean_brightness'] for img in train_imgs]
        contrast = [img['contrast'] for img in train_imgs]
        
        # Plot normal points
        ax2.scatter(bright, contrast, alpha=0.5, c='#4CAF50', label='Training Set')
        
        # Highlight outliers from our detected list
        outlier_filenames = [o['filename'] for o in self.outliers]
        outlier_bright = [img['mean_brightness'] for img in train_imgs if img['filename'] in outlier_filenames]
        outlier_contrast = [img['contrast'] for img in train_imgs if img['filename'] in outlier_filenames]
        
        if outlier_bright:
            ax2.scatter(outlier_bright, outlier_contrast, c='red', s=50, label='Resulting Outliers')
            
        ax2.set_title('Lighting Consistency (Brightness vs Contrast)')
        ax2.set_xlabel('Mean Brightness')
        ax2.set_ylabel('Contrast')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Suspicious Images List
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        outlier_text = "SUSPICIOUS IMAGES (Action Recommended):\n\n"
        
        if self.outliers:
            outlier_text += f"Found {len(self.outliers)} potential outliers that deviate significantly (>2.0σ) from the mean.\n"
            outlier_text += "These images may 'pollute' the PatchCore memory bank assuming 100% normality.\n\n"
            
            # Header
            outlier_text += f"{'FILENAME':<25} | {'ISSUES DETECTED'}\n"
            outlier_text += "-"*80 + "\n"
            
            # List up to 10 outliers
            for i, out in enumerate(self.outliers[:10]):
                reasons = ", ".join(out['outlier_reasons'])
                outlier_text += f"{out['filename']:<25} | {reasons}\n"
            
            if len(self.outliers) > 10:
                outlier_text += f"... and {len(self.outliers)-10} more.\n"
        else:
            outlier_text += "✓ No significant outliers detected. Training set appears consistent.\n"
            outlier_text += "  All images are within 2.0 standard deviations of the mean for all metrics."

        outlier_text += "\n\nPATCHCORE RECOMMENDATION:\n"
        outlier_text += "• Investigate any Red outliers shown above.\n"
        outlier_text += "• If an image is blurry or has different lighting, DELETE it from 'train'.\n"
        outlier_text += "• Outliers reduce model sensitivity by forcing the model to accept 'bad' features as normal."

        ax3.text(0.05, 0.9, outlier_text, ha='left', va='top', fontsize=10,
                family='monospace', transform=ax3.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_advanced_page(self, pdf):
        """Create advanced stability analysis page (Heatmap & Background)"""
        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Advanced Stability Analysis', fontsize=16, fontweight='bold')
        
        # 1. Occupancy Heatmap (Average Image)
        ax1 = fig.add_subplot(gs[0, 0])
        
        if self.avg_train_img is not None:
            # Convert BGR to RGB
            avg_rgb = cv2.cvtColor(self.avg_train_img, cv2.COLOR_BGR2RGB)
            ax1.imshow(avg_rgb)
            ax1.set_title('Average Image (Occupancy Heatmap)')
            ax1.axis('off')
            
            # Add text explanation
            ax1.text(0.5, -0.1, "Sharp edges = Stable alignment\nBlurry edges = Vibration/Movement", 
                    ha='center', transform=ax1.transAxes, fontsize=9)
        else:
            ax1.text(0.5, 0.5, "Not available (insufficient data)", ha='center', va='center')
            ax1.axis('off')
            
        # 2. Pixel Intensity Heatmap (Grayscale)
        ax2 = fig.add_subplot(gs[0, 1])
        
        if self.avg_train_img is not None:
            avg_gray = cv2.cvtColor(self.avg_train_img, cv2.COLOR_BGR2GRAY)
            im = ax2.imshow(avg_gray, cmap='jet')
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            ax2.set_title('Pixel Intensity Distribution')
            ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, "Not available", ha='center', va='center')
            ax2.axis('off')
            
        # 3. Background Noise Analysis
        ax3 = fig.add_subplot(gs[1, :])
        
        if 'bg_noise_mean' in self.stats:
            # Re-calculate corner distribution for plot if possible, 
            # but we didn't store the raw pixels, just stats.
            # We can't plot the histogram unless we store the pixels.
            # Wait, I didn't store 'corners' in self object.
            # I should just report the stats I computed.
            
            # Visualizing background stats
            keys = ['Mean Level', 'Std Dev', 'Max Peak']
            values = [self.stats['bg_noise_mean'], self.stats['bg_noise_std'], self.stats['bg_noise_max']]
            
            bars = ax3.bar(keys, values, color=['#4CAF50', '#2196F3', '#FF9800'])
            ax3.set_ylabel('Pixel Intensity (0-255)')
            ax3.set_title('Background Noise Levels (Target: Near 0)')
            ax3.set_ylim(0, max(30, max(values)*1.2)) # Scale to data but at least 30
            ax3.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            # Add threshold line
            ax3.axhline(y=10, color='r', linestyle='--', label='Noise Threshold (10)')
            ax3.legend()
            
            # Recommendation text
            bg_status = "EXCELLENT" if self.stats['bg_noise_mean'] < 2.0 else "GOOD" if self.stats['bg_noise_mean'] < 5.0 else "WARNING"
            bg_color = "green" if self.stats['bg_noise_mean'] < 2.0 else "orange" if self.stats['bg_noise_mean'] < 5.0 else "red"
            
            info_text = f"""
BACKGROUND ANALYSIS:
• Mean Noise Level: {self.stats['bg_noise_mean']:.2f} / 255
• Noise Status: {bg_status}

Why this matters:
PatchCore detects anomalies by comparing features.
If the black background varies (e.g., dust/lighting), the model
might learn that specific noise patterns are 'normal' or 'abnormal'.
Consistent black background (near 0) ensures focus on the object.
            """
            ax3.text(1.02, 0.5, info_text, transform=ax3.transAxes, va='center',
                    fontsize=10, family='monospace', bbox=dict(facecolor='white', alpha=0.8))
            
        else:
            ax3.text(0.5, 0.5, "Background analysis not available", ha='center')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_conclusion_page(self, pdf):
        """Create final conclusion and recommendations page"""
        fig = plt.figure(figsize=(11, 8.5))
        
        # Calculate Score
        score, penalties = self._calculate_quality_score()
        
        # Determine Grade
        if score >= 90: grade, color = 'A', '#4CAF50'
        elif score >= 80: grade, color = 'B', '#2196F3'
        elif score >= 70: grade, color = 'C', '#FF9800'
        elif score >= 60: grade, color = 'D', '#FF5722'
        else: grade, color = 'F', '#F44336'
        
        # Title
        fig.suptitle('EXECUTIVE SUMMARY & RECOMMENDATIONS', fontsize=20, fontweight='bold', y=0.90)
        
        # 1. Final Grade Circle
        ax_grade = fig.add_axes([0.1, 0.65, 0.2, 0.2])
        circle = plt.Circle((0.5, 0.5), 0.45, color=color, alpha=0.9)
        ax_grade.add_patch(circle)
        ax_grade.text(0.5, 0.5, grade, ha='center', va='center', fontsize=60, color='white', fontweight='bold')
        ax_grade.text(0.5, 0.9, "FINAL GRADE", ha='center', va='center', fontsize=12, fontweight='bold')
        ax_grade.axis('off')
        ax_grade.set_xlim(0, 1)
        ax_grade.set_ylim(0, 1)
        
        # 2. Score Details
        ax_score = fig.add_axes([0.35, 0.65, 0.55, 0.2])
        score_text = f"DATASET QUALITY SCORE: {score}/100\n\n"
        if penalties:
            score_text += "PENALTIES APPLIED:\n" + "\n".join(penalties)
        else:
            score_text += "PERFECT SCORE! No penalties applied."
            
        ax_score.text(0, 0.5, score_text, va='center', fontsize=12, family='monospace')
        ax_score.axis('off')
        
        # 3. Action Plan / Recommendations
        ax_rec = fig.add_axes([0.1, 0.05, 0.8, 0.55])
        
        # Generate dynamic recommendations
        recs = "RECOMMENDED ACTIONS:\n\n"
        
        if score >= 90:
            recs += "✅ READY FOR TRAINING\n"
            recs += "   This dataset is excellent. You can proceed to train PatchCore immediately.\n\n"
        else:
            recs += "⚠️ IMPROVEMENTS NEEDED BEFORE TRAINING\n"
            recs += "   Training with this dataset may result in suboptimal model performance.\n\n"
            
        # Specific based on stats
        if self.stats.get('outlier_count', 0) > 0:
            recs += "1. [CRITICAL] REMOVE OUTLIER IMAGES\n"
            recs += "   Use the 'Clean Outliers' button or manually delete the files listed on Page 5.\n"
            recs += "   These images will teach the model incorrect features.\n\n"
            
        if self.stats.get('centroid_variance', 0) > 2.0:
            recs += "2. [IMPORTANT] IMPROVE ALIGNMENT\n"
            recs += f"   Variance is {self.stats['centroid_variance']:.2f}px. Check for mechanical vibration.\n"
            recs += "   Ensure the camera mount is rigid and the belt speed is constant.\n\n"
            
        if self.stats.get('bg_noise_mean', 0) > 5.0:
            recs += "3. [WARNING] FIX LIGHTING/BACKGROUND\n"
            recs += "   Background is not pure black. Check for ambient light or glare.\n"
            recs += "   Consider adding shrouding or adjusting polarization.\n\n"
            
        if self.stats.get('train_images', 0) < 50:
            recs += "4. [INFO] INCREASE DATASET SIZE\n"
            recs += f"   Only {self.stats['train_images']} images. Aim for at least 50-100 for robust training.\n\n"
            
        recs += "-" * 60 + "\n"
        recs += "PATCHCORE TRAINING CHECKLIST:\n"
        recs += "[ ] No blurry images in train/\n"
        recs += "[ ] No lighting variations (glare/shadows)\n"
        recs += "[ ] Alignment stability verified (< 2px variance)\n"
        recs += "[ ] Background is consistently black"
        
        # Add text box
        ax_rec.text(0, 1, recs, va='top', fontsize=11, family='monospace',
                   bbox=dict(facecolor='#f5f5f5', edgecolor='#ddd', boxstyle='round,pad=1'))
        ax_rec.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_title_page(self, pdf):
        """Create title page"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('DATASET QUALITY REPORT', fontsize=24, fontweight='bold', y=0.85)
        
        # Dataset info
        info_text = f"""
        Dataset Name: {self.stats['dataset_name']}
        
        Analysis Date: {self.stats['analysis_date']}
        
        Total Images: {self.stats['total_images']}
        Training Images: {self.stats['train_images']}
        Debug Images: {self.stats['debug_images']}
        
        Success Rate: {self.stats['success_rate']:.1f}%
        
        
        Generated by InspectionVisionCamera System
        """
        
        plt.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=14,
                family='monospace', transform=fig.transFigure)
        plt.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_overview_page(self, pdf):
        """Create dataset overview page with statistics"""
        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Distribution pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        if self.stats['total_images'] > 0:
            sizes = [self.stats['train_images'], self.stats['debug_images']]
            labels = [f"Training\n{self.stats['train_images']}", f"Debug\n{self.stats['debug_images']}"]
            colors = ['#4CAF50', '#FF9800']
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Image Distribution')
        
        # 2. Success rate gauge
        ax2 = fig.add_subplot(gs[0, 1])
        rate = self.stats['success_rate']
        color = '#4CAF50' if rate >= 70 else '#FF9800' if rate >= 50 else '#F44336'
        ax2.barh([0], [rate], color=color)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('Percentage (%)')
        ax2.set_title('Success Rate')
        ax2.set_yticks([])
        ax2.text(rate/2, 0, f'{rate:.1f}%', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='white')
        
        # 3. Statistics table
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        table_data = [
            ['Metric', 'Training Set', 'Debug Set'],
            ['Count', f"{self.stats['train_images']}", f"{self.stats['debug_images']}"],
            ['Avg Sharpness', f"{self.stats['train_avg_sharpness']:.2f}", f"{self.stats['debug_avg_sharpness']:.2f}"],
            ['Avg Brightness', f"{self.stats['train_avg_brightness']:.2f}", f"{self.stats['debug_avg_brightness']:.2f}"],
            ['Avg Contrast', f"{self.stats['train_avg_contrast']:.2f}", f"{self.stats['debug_avg_contrast']:.2f}"],
        ]
        
        table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax3.set_title('Quality Metrics Summary', fontsize=12, fontweight='bold', pad=20)
        
        # 4. File size distribution
        ax4 = fig.add_subplot(gs[2, :])
        if self.images_info['train'] or self.images_info['debug']:
            train_sizes = [img['size']/1024 for img in self.images_info['train']]  # KB
            debug_sizes = [img['size']/1024 for img in self.images_info['debug']]  # KB
            
            if train_sizes:
                ax4.hist(train_sizes, bins=20, alpha=0.5, label='Training', color='#4CAF50')
            if debug_sizes:
                ax4.hist(debug_sizes, bins=20, alpha=0.5, label='Debug', color='#FF9800')
            
            ax4.set_xlabel('File Size (KB)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('File Size Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_metrics_page(self, pdf):
        """Create quality metrics distributions page"""
        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Quality Metrics Distributions', fontsize=16, fontweight='bold')
        
        # Sharpness distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metric_distribution(ax1, 'sharpness', 'Sharpness Distribution')
        
        # Brightness distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_metric_distribution(ax2, 'mean_brightness', 'Brightness Distribution')
        
        # Contrast distribution
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_metric_distribution(ax3, 'contrast', 'Contrast Distribution')
        
        # Box plot comparison
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_boxplot_comparison(ax4)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_distribution(self, ax, metric_key, title):
        """Plot distribution of a specific metric"""
        train_values = [img[metric_key] for img in self.images_info['train']]
        debug_values = [img[metric_key] for img in self.images_info['debug']]
        
        if train_values:
            ax.hist(train_values, bins=20, alpha=0.6, label='Training', color='#4CAF50')
        if debug_values:
            ax.hist(debug_values, bins=20, alpha=0.6, label='Debug', color='#FF9800')
        
        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_boxplot_comparison(self, ax):
        """Create box plot comparison of sharpness"""
        train_sharpness = [img['sharpness'] for img in self.images_info['train']]
        debug_sharpness = [img['sharpness'] for img in self.images_info['debug']]
        
        data = []
        labels = []
        
        if train_sharpness:
            data.append(train_sharpness)
            labels.append('Training')
        if debug_sharpness:
            data.append(debug_sharpness)
            labels.append('Debug')
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = ['#4CAF50', '#FF9800']
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax.set_title('Sharpness Comparison')
        ax.set_ylabel('Sharpness Value')
        ax.grid(True, alpha=0.3)
    
    def _create_alignment_page(self, pdf):
        """Create alignment verification page (critical for PatchCore)"""
        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Alignment Verification (PatchCore Critical)', fontsize=16, fontweight='bold')
        
        # Get training data
        train_images = self.images_info['train']
        
        if len(train_images) > 1:
            centroids = [img['centroid'] for img in train_images]
            diameters = [img['diameter'] for img in train_images if img['diameter'] > 0]
            
            # 1. Centroid scatter plot
            ax1 = fig.add_subplot(gs[0, :])
            if centroids:
                xs = [c[0] for c in centroids]
                ys = [c[1] for c in centroids]
                
                ax1.scatter(xs, ys, c='#4CAF50', s=100, alpha=0.6, edgecolors='black')
                ax1.axhline(y=np.mean(ys), color='r', linestyle='--', label=f'Mean Y: {np.mean(ys):.1f}')
                ax1.axvline(x=np.mean(xs), color='b', linestyle='--', label=f'Mean X: {np.mean(xs):.1f}')
                
                # Draw std deviation ellipse
                std_x = self.stats['centroid_std_x']
                std_y = self.stats['centroid_std_y']
                
                from matplotlib.patches import Ellipse
                ellipse = Ellipse((np.mean(xs), np.mean(ys)), 
                                width=std_x*2, height=std_y*2, 
                                fill=False, edgecolor='orange', linewidth=2, 
                                label=f'±1σ ({std_x:.1f}px, {std_y:.1f}px)')
                ax1.add_patch(ellipse)
                
                ax1.set_xlabel('X Position (pixels)')
                ax1.set_ylabel('Y Position (pixels)')
                ax1.set_title(f'Centroid Stability (Variance: {self.stats["centroid_variance"]:.2f}px)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_aspect('equal', adjustable='box')
            
            # 2. Diameter histogram
            ax2 = fig.add_subplot(gs[1, 0])
            if diameters:
                ax2.hist(diameters, bins=20, color='#4CAF50', alpha=0.7, edgecolor='black')
                ax2.axvline(x=self.stats['diameter_mean'], color='r', linestyle='--', 
                          label=f'Mean: {self.stats["diameter_mean"]:.1f}px')
                ax2.axvline(x=self.stats['diameter_mean']-self.stats['diameter_std'], 
                          color='orange', linestyle=':', label=f'±1σ: {self.stats["diameter_std"]:.1f}px')
                ax2.axvline(x=self.stats['diameter_mean']+self.stats['diameter_std'], 
                          color='orange', linestyle=':')
                
                ax2.set_xlabel('Diameter (pixels)')
                ax2.set_ylabel('Frequency')
                ax2.set_title(f'Scale Stability (CV: {self.stats["diameter_cv"]:.2f}%)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. Diameter over sequence
            ax3 = fig.add_subplot(gs[1, 1])
            if diameters:
                ax3.plot(range(len(diameters)), diameters, 'o-', color='#4CAF50', markersize=6)
                ax3.axhline(y=self.stats['diameter_mean'], color='r', linestyle='--', 
                          label=f'Mean: {self.stats["diameter_mean"]:.1f}px')
                ax3.fill_between(range(len(diameters)), 
                               self.stats['diameter_mean']-self.stats['diameter_std'],
                               self.stats['diameter_mean']+self.stats['diameter_std'],
                               alpha=0.2, color='orange', label=f'±1σ')
                
                ax3.set_xlabel('Image Index')
                ax3.set_ylabel('Diameter (pixels)')
                ax3.set_title('Scale Consistency Over Sequence')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 4. Alignment metrics table and recommendations
            ax4 = fig.add_subplot(gs[2, :])
            ax4.axis('off')
            
            # Determine recommendations
            centroid_status = '✓ GOOD' if self.stats['centroid_variance'] < 2.0 else '⚠ MODERATE' if self.stats['centroid_variance'] < 5.0 else '✗ POOR'
            centroid_color = '#4CAF50' if self.stats['centroid_variance'] < 2.0 else '#FF9800' if self.stats['centroid_variance'] < 5.0 else '#F44336'
            
            scale_status = '✓ GOOD' if self.stats['diameter_cv'] < 2.0 else '⚠ MODERATE' if self.stats['diameter_cv'] < 5.0 else '✗ POOR'
            scale_color = '#4CAF50' if self.stats['diameter_cv'] < 2.0 else '#FF9800' if self.stats['diameter_cv'] < 5.0 else '#F44336'
            
            metrics_text = f"""
ALIGNMENT METRICS (PatchCore Sensitivity Analysis):

Centroid Variance: {self.stats['centroid_variance']:.2f} pixels - {centroid_status}
  • X Std Dev: {self.stats['centroid_std_x']:.2f}px  |  Y Std Dev: {self.stats['centroid_std_y']:.2f}px
  • Recommendation: < 2.0px excellent, < 5.0px acceptable, > 5.0px may cause false positives

Scale Stability: CV = {self.stats['diameter_cv']:.2f}% - {scale_status}
  • Mean Diameter: {self.stats['diameter_mean']:.1f}px  |  Std Dev: {self.stats['diameter_std']:.1f}px
  • Recommendation: < 2% excellent, < 5% acceptable, > 5% may affect detection accuracy

PATCHCORE RECOMMENDATIONS:
✓ Excellent alignment (< 2px variance) ensures stable patch matching
✓ Consistent scale (< 2% CV) prevents false positives at object edges
⚠ High variance may require data augmentation or improved alignment pipeline
            """
            
            ax4.text(0.05, 0.5, metrics_text, ha='left', va='center', fontsize=10,
                    family='monospace', transform=ax4.transAxes)
        
        else:
            # Not enough data
            ax_text = fig.add_subplot(gs[:, :])
            ax_text.text(0.5, 0.5, 'Insufficient training images for alignment analysis\n(Need at least 2 images)',
                        ha='center', va='center', fontsize=14)
            ax_text.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_quality_analysis_page(self, pdf):
        """Create detailed quality analysis page"""
        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)
        
        fig.suptitle('Image Quality Analysis', fontsize=16, fontweight='bold')
        
        # Scatter plot: Sharpness vs Brightness
        ax1 = fig.add_subplot(gs[0, 0])
        
        if self.images_info['train']:
            train_sharp = [img['sharpness'] for img in self.images_info['train']]
            train_bright = [img['mean_brightness'] for img in self.images_info['train']]
            ax1.scatter(train_sharp, train_bright, alpha=0.5, s=50, c='#4CAF50', label='Training')
        
        if self.images_info['debug']:
            debug_sharp = [img['sharpness'] for img in self.images_info['debug']]
            debug_bright = [img['mean_brightness'] for img in self.images_info['debug']]
            ax1.scatter(debug_sharp, debug_bright, alpha=0.5, s=50, c='#FF9800', label='Debug')
        
        ax1.set_xlabel('Sharpness')
        ax1.set_ylabel('Mean Brightness')
        ax1.set_title('Sharpness vs Brightness')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot: Sharpness vs Contrast
        ax2 = fig.add_subplot(gs[1, 0])
        
        if self.images_info['train']:
            train_sharp = [img['sharpness'] for img in self.images_info['train']]
            train_contrast = [img['contrast'] for img in self.images_info['train']]
            ax2.scatter(train_sharp, train_contrast, alpha=0.5, s=50, c='#4CAF50', label='Training')
        
        if self.images_info['debug']:
            debug_sharp = [img['sharpness'] for img in self.images_info['debug']]
            debug_contrast = [img['contrast'] for img in self.images_info['debug']]
            ax2.scatter(debug_sharp, debug_contrast, alpha=0.5, s=50, c='#FF9800', label='Debug')
        
        ax2.set_xlabel('Sharpness')
        ax2.set_ylabel('Contrast')
        ax2.set_title('Sharpness vs Contrast')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_samples_page(self, pdf):
        """Create page with sample images"""
        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
        
        fig.suptitle('Sample Images', fontsize=16, fontweight='bold')
        
        # Show up to 6 training samples and 6 debug samples
        train_samples = self.images_info['train'][:6]
        debug_samples = self.images_info['debug'][:6]
        
        # Training samples (top 2 rows)
        for idx, img_info in enumerate(train_samples):
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            img = cv2.imread(img_info['path'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"Train: {img_info['filename'][:15]}...", fontsize=8)
            ax.axis('off')
        
        # Debug samples (bottom row + any remaining)
        for idx, img_info in enumerate(debug_samples):
            row = 2
            col = idx
            if col >= 4:
                break
            ax = fig.add_subplot(gs[row, col])
            
            img = cv2.imread(img_info['path'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"Debug: {img_info['filename'][:15]}...", fontsize=8)
            ax.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def generate_dataset_report(dataset_folder, output_path=None):
    """
    Generate a comprehensive dataset report
    
    Args:
        dataset_folder: Path to dataset folder
        output_path: Optional output path for PDF report
    
    Returns:
        Path to generated report
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(dataset_folder, f'report_{timestamp}.pdf')
    
    generator = DatasetReportGenerator(dataset_folder)
    generator.analyze_dataset()
    generator.generate_report(output_path)
    
    return output_path
