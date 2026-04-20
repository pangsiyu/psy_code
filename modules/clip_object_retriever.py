import os
import json
import torch
from PIL import Image
from typing import List, Dict, Tuple
from modelscope import AutoModel, AutoTokenizer

class CLIPObjectRetriever:
    def __init__(self, scene_image_dir: str, cache_dir: str = "./checkpoints"):
        """
        初始化 CLIP 目标检索器
        :param scene_image_dir: 场景离散 RGB 图像库目录
        :param cache_dir: 模型缓存目录
        """
        self.scene_image_dir = scene_image_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_file = os.path.join(scene_image_dir, "image_features_cache.pt")
        
        # 加载 CLIP 模型（轻量且适合中文-图像检索）
        print("[CLIP 检索] 加载 CLIP 模型...")
        self.clip_model = AutoModel.from_pretrained(
            'damo/multi-modal_clip-vit-large-patch14',
            trust_remote_code=True,
            cache_dir=cache_dir
        ).to(self.device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            'damo/multi-modal_clip-vit-large-patch14',
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        self.clip_model.eval()
        
        # 加载/提取场景图像库的特征缓存
        self.image_list = []
        self.image_features = None
        self._load_or_extract_features()

    def _extract_image_feature(self, image_path: str) -> torch.Tensor:
        """提取单张图像的 CLIP 特征"""
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        
        # CLIP 图像预处理（简化版）
        img_tensor = torch.tensor([img]).permute(0, 3, 1, 2).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            # 注意：这里需根据实际使用的 CLIP 模型 API 调整
            # 以下为示例，具体以 ModelScope 文档为准
            features = self.clip_model.get_image_features(img_tensor)
        return features.squeeze().cpu()

    def _load_or_extract_features(self):
        """加载缓存的特征，或提取并缓存"""
        self.image_list = [f for f in os.listdir(self.scene_image_dir) if f.endswith((".jpg", ".png"))]
        
        if os.path.exists(self.cache_file):
            print(f"[CLIP 检索] 加载图像特征缓存：{self.cache_file}")
            cache_data = torch.load(self.cache_file)
            self.image_features = cache_data["features"]
            # 验证图像列表是否匹配
            if set(cache_data["image_list"]) != set(self.image_list):
                print("[CLIP 检索] 警告：图像库已更新，重新提取特征")
                self._extract_and_cache()
        else:
            print("[CLIP 检索] 提取场景图像库特征...")
            self._extract_and_cache()

    def _extract_and_cache(self):
        """提取所有图像特征并缓存"""
        features = []
        for img_name in self.image_list:
            img_path = os.path.join(self.scene_image_dir, img_name)
            feat = self._extract_image_feature(img_path)
            features.append(feat)
            print(f"[CLIP 检索] 已提取：{img_name}")
        
        self.image_features = torch.stack(features)
        torch.save({
            "image_list": self.image_list,
            "features": self.image_features
        }, self.cache_file)
        print(f"[CLIP 检索] 特征缓存已保存：{self.cache_file}")

    def retrieve(self, object_name: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        检索包含目标物体的图像
        :param object_name: 目标物体名称，如"红丝绒长裙"
        :param top_k: 返回前 k 个最匹配的图像
        :return: [(图像ID, 相似度分数), ...]
        """
        print(f"[CLIP 检索] 正在检索目标：{object_name}")
        
        # 提取文本特征
        text_input = f"一张包含{object_name}的照片"
        with torch.no_grad():
            # 注意：这里需根据实际使用的 CLIP 模型 API 调整
            text_features = self.clip_model.get_text_features(self.clip_tokenizer(text_input)).squeeze().cpu()
        
        # 计算相似度
        similarities = torch.nn.functional.cosine_similarity(
            text_features.unsqueeze(0),
            self.image_features
        )
        
        # 获取 top_k 结果
        top_indices = torch.topk(similarities, top_k).indices
        results = []
        for idx in top_indices:
            img_name = self.image_list[idx]
            score = similarities[idx].item()
            results.append((img_name, score))
            print(f"[CLIP 检索] 匹配结果：{img_name} (相似度：{score:.4f})")
        
        return results