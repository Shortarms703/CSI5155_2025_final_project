import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

def extract_features(image_path):
    """Extract Hu moments from image"""
    img = cv2.imread(str(image_path), 0)
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)
    features = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return features.flatten()

def cluster_polygon_type(polygon_folder, output_base, reduction_percent=0.10):
    """Cluster images for a single polygon type"""
    polygon_name = polygon_folder.name
    
    image_paths = list(polygon_folder.glob("*.png"))
    n_images = len(image_paths)
    
    if n_images == 0:
        print(f"  No images found in {polygon_folder}")
        return None
    
    n_clusters = max(1, int(n_images * reduction_percent))
    
    print(f"\n{polygon_name}: {n_images} images → {n_clusters} clusters")
    
    print(f"  Extracting features...")
    features = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            feat = extract_features(img_path)
            features.append(feat)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"  Skipped {img_path.name}: {e}")
    
    features = np.array(features)
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # PCA
    n_components = min(5, features_normalized.shape[1], len(features) - 1)
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_normalized)
    
    # Kmeans
    print(f"  Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_pca)
    
    cluster_data = {
        "polygon_type": polygon_name,
        "total_images": n_images,
        "num_clusters": n_clusters,
        "reduction_percent": reduction_percent,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "pca_total_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "pca_components": n_components,
        "clusters": []
    }
    
    for cluster_id in range(n_clusters):
        cluster_images = []
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        cluster_features = features_pca[cluster_mask]
        centroid = kmeans.cluster_centers_[cluster_id].tolist()
        
        distances = np.linalg.norm(cluster_features - kmeans.cluster_centers_[cluster_id], axis=1)
        
        representative_idx = int(distances.argmin())
        
        for idx, img_idx in enumerate(cluster_indices):
            img_path = valid_paths[img_idx]
            
            cluster_images.append({
                "image_path": str(img_path),
                "filename": img_path.name,
                "distance_to_centroid": float(distances[idx]),
                "is_representative": bool(idx == representative_idx)
            })
        
        cluster_images.sort(key=lambda x: x["distance_to_centroid"])
        
        cluster_info = {
            "cluster_id": int(cluster_id),
            "cluster_name": f"cluster_{cluster_id:03d}",
            "size": int(cluster_mask.sum()),
            "centroid": centroid,
            "mean_distance_to_centroid": float(distances.mean()),
            "max_distance_to_centroid": float(distances.max()),
            "min_distance_to_centroid": float(distances.min()),
            "representative_image": str(valid_paths[cluster_indices[representative_idx]]),
            "images": cluster_images
        }
        
        cluster_data["clusters"].append(cluster_info)
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    json_output_path = output_base / f"{polygon_name}_clustering.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved: {json_output_path}")
    
    return cluster_data

def main():
    raster_folder = "64x64/raster"
    output_folder = "clusterization"
    reduction_percent = 0.10
    
    raster_path = Path(raster_folder)
    polygon_folders = sorted(raster_path.glob("polygon_*"))
    
    if not polygon_folders:
        print(f"No polygon_* folders found in {raster_folder}")
        return
    
    print(f"Found {len(polygon_folders)} polygon types")
    print(f"Target: {reduction_percent*100:.0f}% clusters (10% of images)")
    
    output_base = Path(output_folder)
    output_base.mkdir(exist_ok=True)
    
    all_results = {
        "project_info": {
            "source_folder": raster_folder,
            "output_folder": output_folder,
            "reduction_percent": reduction_percent,
            "total_polygon_types": len(polygon_folders)
        },
        "polygon_files": []
    }
    
    for polygon_folder in polygon_folders:
        result = cluster_polygon_type(polygon_folder, output_base, reduction_percent)
        if result:
            all_results["polygon_files"].append(f"{polygon_folder.name}_clustering.json")
    
    summary_path = output_base / "clustering_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Done! All JSON files saved to '{output_folder}'")
    print(f"✓ Summary saved to '{summary_path}'")
    print(f"\nGenerated files:")
    print(f"  {output_folder}/")
    print(f"    clustering_summary.json")
    for pf in all_results["polygon_files"]:
        print(f"    {pf}")

if __name__ == "__main__":
    main()
