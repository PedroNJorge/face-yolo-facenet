# face-yolo-facenet

# Worflow

```mermaid
graph TD
    A[Start] --> B[Detect Face]
    B --> C{In DB?}
    C -->|Yes| D[Update Last Seen]
    C -->|No| E[Ask User: Known Person?]
    E -->|Yes| F[Add Metadata + Save to Known Faces]
    E -->|No| G[Save to Unknown Faces + Cluster]
```

# Schemas Venn Diagrams

```mermaid
%% Face Recognition Schema Relationships
classDiagram
    %% Core Entities
    class FaceEmbedding {
        +str embedding (Base85-LZ4)
        +str encoding_version
        +from_tensor(torch.Tensor)$ FaceEmbedding
        +to_tensor() torch.Tensor
    }

    class FaceImageMetadata {
        +str image_hash
        +str original_path
        +float timestamp
        +float detection_confidence
        +tuple face_bbox
    }

    class FaceImageRecord {
        +FaceImageMetadata metadata
        +FaceEmbedding embedding
        +get_embedding_tensor() torch.Tensor
    }

    class FaceMetadata {
        +str person_id
        +str name
        +float first_seen
        +float last_updated
        +List[str] source_images
    }

    class FaceProfile {
        +str person_id
        +FaceMetadata metadata
        +FaceEmbedding main_embedding
        +Dict[str, FaceEmbedding] embeddings
        +update_main_embedding(tensor, weight=0.2)
        +add_embedding(image_hash, tensor)
    }

    %% Relationships
    FaceImageRecord *-- FaceImageMetadata
    FaceImageRecord *-- FaceEmbedding
    FaceProfile *-- FaceMetadata
    FaceProfile *-- FaceEmbedding
    FaceProfile o-- FaceEmbedding : via embeddings

    %% Style Annotations
    note for FaceEmbedding "Secured Storage:\n1. Tensor → pickle\n2. LZ4 compress\n3. Base85 encode"
    note for FaceProfile "Central Entity:\n- Manages weighted average\n- Tracks all embeddings"
```

# References
YOLO Model by AdamCodd: https://huggingface.co/AdamCodd/YOLOv11n-face-detection </br>
Pytorch's FaceNet repo: https://github.com/timesler/facenet-pytorch
