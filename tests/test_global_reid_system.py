import pytest
import numpy as np
from app.reid.identity import GlobalIdentity
from app.reid.global_gallery import GlobalGallery
from app.reid.matcher import GlobalMatcher
from app.reid.embedding_bank import TrackEmbeddingBank
from app.reid.extractors.pose_extractor import PoseExtractor
from app.reid.extractors.par_extractor import ParExtractor

def create_synthetic_embedding(base_seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(base_seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)

def create_related_embedding(base_vec: np.ndarray, target_similarity: float, noise_seed: int = 99) -> np.ndarray:
    """Creates a unit vector having approximately `target_similarity` cosine similarity with base_vec."""
    rng = np.random.RandomState(noise_seed)
    noise = rng.randn(len(base_vec)).astype(np.float32)
    # Orthogonalize noise with respect to base_vec
    noise = noise - np.dot(noise, base_vec) * base_vec
    noise = noise / np.linalg.norm(noise)
    
    # Linear combination
    theta = np.arccos(np.clip(target_similarity, -1.0, 1.0))
    res = np.cos(theta) * base_vec + np.sin(theta) * noise
    return (res / np.linalg.norm(res)).astype(np.float32)

@pytest.fixture
def matcher_config():
    return {
        "weight_reid": 0.60,
        "weight_par": 0.25,
        "weight_context": 0.15,
        "threshold_match": 0.80,
        "threshold_ambiguous": 0.68,
        "threshold_cross_view": 0.62,
        "reid_ambiguous_margin": 0.04,
        "reid_confirm_frames": 3,
        "pending_confirm_frames": 3,
        "camera_transitions": {
            "cctv1": ["cctv2", "cctv3"],
            "cctv2": ["cctv1", "cctv3"],
            "waiting_lobby": ["pantry", "passage_1"],
            "pantry": ["waiting_lobby", "passage_1"],
            "passage_1": ["waiting_lobby", "pantry"]
        }
    }

def test_scenario_1_cctv1_back_to_cctv2_side_to_cctv2_front(matcher_config, tmp_path):
    """
    Scenario 1:
    CCTV 1: Person 1 observed ONLY from BACK (B1)
    CCTV 2: Person 1 enters, observed first from SIDE (S1)
    CCTV 2: Person 1 turns, observed from FRONT (F1)
    Expected: All observations must remain Global_ID = Person_001.
    """
    storage = str(tmp_path / "test_gallery.pkl")
    gallery = GlobalGallery(storage_path=storage, clear_on_start=True)
    matcher = GlobalMatcher(gallery=gallery, config=matcher_config)

    # 1. Base representations for Person 1
    # B1 = Back view embedding
    # S1 = Side view embedding (~0.62 similarity to B1, realistic cross-view drop)
    # F1 = Front view embedding (~0.88 similarity to S1, ~0.60 to B1)
    b1 = create_synthetic_embedding(base_seed=101)
    s1 = create_related_embedding(b1, target_similarity=0.62, noise_seed=102)
    f1 = create_related_embedding(s1, target_similarity=0.88, noise_seed=103)

    par_blue_black = {
        "upper_color": "blue",
        "lower_color": "black",
        "features": np.zeros(256, dtype=np.float32)
    }

    # Step A: CCTV1 ONLY BACK VIEW
    cctv1_embs = [{
        "embedding": b1,
        "viewpoint": "BACK",
        "quality": 0.85,
        "camera_id": "cctv1",
        "track_id": 5,
        "par_attributes": par_blue_black
    }]
    gid1 = gallery.add_identity([{"camera_id": "cctv1", "track_id": 5}], cctv1_embs, status="CONFIRMED")
    assert gid1 == "Person_001"
    
    # Verify Person_001 has only BACK view
    ident1 = gallery.get_identities()["Person_001"]
    assert ident1.has_viewpoint("BACK") is True
    assert ident1.has_viewpoint("SIDE") is False
    assert ident1.has_viewpoint("FRONT") is False

    # Step B: CCTV2 SIDE VIEW (Cross-view matching!)
    cctv2_side_embs = [{
        "embedding": s1,
        "viewpoint": "SIDE",
        "quality": 0.82,
        "camera_id": "cctv2",
        "track_id": 18,
        "par_attributes": par_blue_black
    }]
    best_gid, sim, sec_sim, status = matcher.match(
        cctv2_side_embs,
        current_camera="cctv2",
        track_id=18,
        consecutive_candidate_count=1
    )
    # Must pick Person_001 as candidate
    assert best_gid == "Person_001"
    # Status should be PENDING or MATCH, NEVER NO_MATCH!
    assert status in ["PENDING", "MATCH"]
    
    # Step C: CCTV2 FRONT VIEW appears
    # Now the track has both SIDE and FRONT observations
    cctv2_combined_embs = [
        *cctv2_side_embs,
        {
            "embedding": f1,
            "viewpoint": "FRONT",
            "quality": 0.90,
            "camera_id": "cctv2",
            "track_id": 18,
            "par_attributes": par_blue_black
        }
    ]
    best_gid2, sim2, sec_sim2, status2 = matcher.match(
        cctv2_combined_embs,
        current_camera="cctv2",
        track_id=18,
        consecutive_candidate_count=3
    )
    assert best_gid2 == "Person_001"
    
    # Confirm the identity in gallery
    gallery.update_identity(best_gid2, [{"camera_id": "cctv2", "track_id": 18}], cctv2_combined_embs, status="CONFIRMED")
    
    # Verify Person_001 now possesses BACK, SIDE, and FRONT
    updated_ident = gallery.get_identities()["Person_001"]
    assert updated_ident.has_viewpoint("BACK") is True
    assert updated_ident.has_viewpoint("SIDE") is True
    assert updated_ident.has_viewpoint("FRONT") is True
    assert len(gallery.get_identities()) == 1, "Must NOT spawn Person_002 or Person_003!"

def test_scenario_2_cctv1_back_to_cctv2_front_directly(matcher_config, tmp_path):
    """
    Scenario 2:
    CCTV 1: Person 1 observed from BACK
    CCTV 2: Person 1 observed directly from FRONT (no side view)
    Expected: Same Global ID (Person_001).
    """
    storage = str(tmp_path / "test_gallery_2.pkl")
    gallery = GlobalGallery(storage_path=storage, clear_on_start=True)
    matcher = GlobalMatcher(gallery=gallery, config=matcher_config)

    b1 = create_synthetic_embedding(base_seed=201)
    # Direct front view with typical cross-view similarity ~0.60
    f1 = create_related_embedding(b1, target_similarity=0.60, noise_seed=202)
    par_data = {"upper_color": "white", "lower_color": "blue"}

    # CCTV1 BACK
    gallery.add_identity(
        [{"camera_id": "cctv1", "track_id": 1}],
        [{"embedding": b1, "viewpoint": "BACK", "quality": 0.85, "camera_id": "cctv1", "par_attributes": par_data}]
    )

    # CCTV2 FRONT
    cctv2_front = [{
        "embedding": f1,
        "viewpoint": "FRONT",
        "quality": 0.88,
        "camera_id": "cctv2",
        "track_id": 10,
        "par_attributes": par_data
    }]
    best_gid, sim, sec_sim, status = matcher.match(
        cctv2_front,
        current_camera="cctv2",
        track_id=10,
        consecutive_candidate_count=2
    )
    assert best_gid == "Person_001"
    assert status in ["PENDING", "MATCH"]

def test_scenario_3_cctv1_back_cctv2_side_occlusion_cctv2_front(matcher_config, tmp_path):
    """
    Scenario 3:
    CCTV1 BACK -> CCTV2 SIDE -> Occlusion -> CCTV2 FRONT
    Expected: Same Global ID maintained across occlusion.
    """
    storage = str(tmp_path / "test_gallery_3.pkl")
    gallery = GlobalGallery(storage_path=storage, clear_on_start=True)
    matcher = GlobalMatcher(gallery=gallery, config=matcher_config)

    b1 = create_synthetic_embedding(base_seed=301)
    s1 = create_related_embedding(b1, target_similarity=0.62, noise_seed=302)
    f1 = create_related_embedding(s1, target_similarity=0.88, noise_seed=303)
    par_data = {"upper_color": "grey", "lower_color": "black"}

    # CCTV1 BACK
    gallery.add_identity(
        [{"camera_id": "cctv1", "track_id": 1}],
        [{"embedding": b1, "viewpoint": "BACK", "quality": 0.85, "camera_id": "cctv1", "par_attributes": par_data}]
    )

    # CCTV2 SIDE (track 12)
    side_embs = [{"embedding": s1, "viewpoint": "SIDE", "quality": 0.82, "camera_id": "cctv2", "par_attributes": par_data}]
    gid_side, _, _, status_side = matcher.match(side_embs, current_camera="cctv2", track_id=12)
    assert gid_side == "Person_001"

    # Occlusion occurs; track 12 lost. Later, track 15 appears with FRONT view (f1)
    front_embs = [{"embedding": f1, "viewpoint": "FRONT", "quality": 0.89, "camera_id": "cctv2", "par_attributes": par_data}]
    gid_front, sim_front, _, status_front = matcher.match(
        front_embs,
        current_camera="cctv2",
        track_id=15,
        consecutive_candidate_count=3
    )
    assert gid_front == "Person_001"

def test_scenario_4_two_people_same_clothing_color(matcher_config, tmp_path):
    """
    Scenario 4:
    Person 1 and Person 2 both wear: blue shirt, black pants.
    The system MUST NOT match Person 2 to Person 1 based on clothing color alone.
    OSNet primary identity feature must enforce separation!
    """
    storage = str(tmp_path / "test_gallery_4.pkl")
    gallery = GlobalGallery(storage_path=storage, clear_on_start=True)
    matcher = GlobalMatcher(gallery=gallery, config=matcher_config)

    p1_emb = create_synthetic_embedding(base_seed=401)
    # Distinct person with low ReID similarity (~0.30)
    p2_emb = create_related_embedding(p1_emb, target_similarity=0.30, noise_seed=402)

    # Both have the exact same clothing colors!
    same_clothing = {"upper_color": "blue", "lower_color": "black"}

    # Register Person 1
    gallery.add_identity(
        [{"camera_id": "cctv1", "track_id": 1}],
        [{"embedding": p1_emb, "viewpoint": "FRONT", "quality": 0.90, "camera_id": "cctv1", "par_attributes": same_clothing}]
    )

    # Person 2 enters
    p2_query = [{
        "embedding": p2_emb,
        "viewpoint": "FRONT",
        "quality": 0.88,
        "camera_id": "cctv1",
        "track_id": 2,
        "par_attributes": same_clothing
    }]
    best_gid, sim, sec_sim, status = matcher.match(p2_query, current_camera="cctv1", track_id=2)
    
    # Must NOT confirm Person 1! Anti-veto safeguard must reject due to low ReID score
    assert status == "NO_MATCH", f"Person 2 with distinct embedding was incorrectly assigned {best_gid} (status={status}, sim={sim})"

def test_scenario_5_person_leaves_cctv1_and_reenters_cctv2(matcher_config, tmp_path):
    """
    Scenario 5:
    Person leaves CCTV1, transit time elapses, person re-enters in CCTV2.
    Expected: Same Global ID is recognized and maintained.
    """
    storage = str(tmp_path / "test_gallery_5.pkl")
    gallery = GlobalGallery(storage_path=storage, clear_on_start=True)
    matcher = GlobalMatcher(gallery=gallery, config=matcher_config)

    emb = create_synthetic_embedding(base_seed=501)
    par_data = {"upper_color": "red", "lower_color": "black"}

    # Observed in CCTV1
    gallery.add_identity(
        [{"camera_id": "waiting_lobby", "track_id": 4}],
        [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.88, "camera_id": "waiting_lobby", "par_attributes": par_data}]
    )

    # Re-enters in CCTV2 (pantry)
    query_reenter = [{
        "embedding": emb,
        "viewpoint": "FRONT",
        "quality": 0.86,
        "camera_id": "pantry",
        "track_id": 9,
        "par_attributes": par_data
    }]
    best_gid, sim, sec_sim, status = matcher.match(query_reenter, current_camera="pantry", track_id=9)
    assert best_gid == "Person_001"
    assert status == "MATCH"

def test_scenario_6_person_leaves_both_cameras_and_later_reappears(matcher_config, tmp_path):
    """
    Scenario 6:
    Person leaves all cameras for an extended period, then reappears.
    Identity bank recovers identity without ID fragmentation.
    """
    storage = str(tmp_path / "test_gallery_6.pkl")
    gallery = GlobalGallery(storage_path=storage, clear_on_start=True)
    matcher = GlobalMatcher(gallery=gallery, config=matcher_config)

    p1_front = create_synthetic_embedding(base_seed=601)
    p1_back = create_related_embedding(p1_front, target_similarity=0.62, noise_seed=602)
    par_data = {"upper_color": "green", "lower_color": "blue"}

    # Person 1 was established with front and back views
    gallery.add_identity(
        [{"camera_id": "cctv1", "track_id": 1}],
        [
            {"embedding": p1_front, "viewpoint": "FRONT", "quality": 0.90, "camera_id": "cctv1", "par_attributes": par_data},
            {"embedding": p1_back, "viewpoint": "BACK", "quality": 0.85, "camera_id": "cctv1", "par_attributes": par_data}
        ]
    )

    # Person 1 reappears much later in CCTV2 seen from the back
    reappear_query = [{
        "embedding": p1_back,
        "viewpoint": "BACK",
        "quality": 0.88,
        "camera_id": "cctv2",
        "track_id": 33,
        "par_attributes": par_data
    }]
    best_gid, sim, _, status = matcher.match(reappear_query, current_camera="cctv2", track_id=33)
    assert best_gid == "Person_001"
    assert status == "MATCH"
    assert len(gallery.get_identities()) == 1
