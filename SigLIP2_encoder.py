import torch
import torchvision
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
import joblib

class SigLIP2Network:
    # def __init__(self, device="cuda", *, ckpt="google/siglip2-base-patch16-512"):
    def __init__(self, device="cuda", *, ckpt="google/siglip2-so400m-patch16-512"):
        # Load the SigLIP model and processor from Transformers
        self.model = AutoModel.from_pretrained(
            ckpt,
            torch_dtype=torch.float16
        ).to(device)
        self.model.eval()  # Set the model to evaluation mode, as in the original code
        
        self.processor = AutoProcessor.from_pretrained(ckpt)
        
        self.clip_n_dims = 1024
        
        # Initialize positive and negative phrases (unchanged)
        self.negatives = ("object", "things", "stuff", "texture")
        self.positives = (" ",)

        self.pca = joblib.load("./pca_model.joblib")
        self.rpj = joblib.load("./rpj.joblib")
        
        # Compute initial embeddings for positives and negatives
        with torch.no_grad():
            # Tokenize and compute positive embeddings
            pos_inputs = self.processor(
                text=self.positives, padding="max_length", max_length=64, return_tensors="pt"
            ).to(device)
            self.pos_embeds = self.model.get_text_features(**pos_inputs)
            
            # Tokenize and compute negative embeddings
            neg_inputs = self.processor(
                text=self.negatives, padding="max_length", max_length=64, return_tensors="pt"
            ).to(device)
            self.neg_embeds = self.model.get_text_features(**neg_inputs)
        
        # Normalize embeddings (unchanged logic)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        # print(self.negatives, self.neg_embeds.shape)
        p = phrases_embeds.to(embed.dtype)
        # print(embed.shape, p.shape)
        # p = self.pca.transform(p.cpu().numpy()) # (11, 1152) -> (11, 512)
        p = self.rpj.transform(p.cpu().numpy()) # (11, 1152) -> (11, 512)
        p = torch.tensor(p).to(embed.device)
        # print(p.shape)

        output = torch.mm(embed, p.T) # 721240x512 * 512x11 -> 721240x11
        positive_vals = output[..., positive_id : positive_id + 1] # (721240, 1) similarities between the rendered embeddings and the positive query phrase
        negative_vals = output[..., len(self.positives) :] # (721240, 4) similarities between the rendered embeddings and the negative query phrases [object, things,...]
        repeated_pos = positive_vals.repeat(1, len(self.negatives)) # (721240, 1) -> (721240, 4)

        sims = torch.stack((repeated_pos, negative_vals), dim=-1) # torch.Size([721240, 4, 2])
        softmax = torch.softmax(10 * sims, dim=-1) # torch.Size([721240, 4, 2])
        best_id = softmax[..., 0].argmin(dim=1) # torch.Size([721240])

        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    @torch.no_grad()
    def encode_image(self, input, mask=None):
        input = input.to("cuda")
        inp = self.processor(
                images=input, return_tensors="pt"
        # ).half().to(input.device)
        ).to("cuda")
        # print(input)

        # inp.pixel_values = inp.pixel_values.half().to("cuda")


        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        # print(type(input), input.device, type(inp))
        # print("pix_val")
        # for i in inp:
        #     print(i, inp[i].device, inp[i].dtype)
        # print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # return self.model.get_image_features(**inp).half().to("cuda")
        return self.model.get_image_features(**inp).to("cuda")

    @torch.no_grad()
    def encode_text(self, text_list, device):
        text = self.processor(
                text=text_list, padding="max_length", max_length=64, return_tensors="pt"
        ).to("cuda")
        return self.model.get_text_features(**text).to(device)
    
    @torch.no_grad()
    def set_positives(self, text_list):
        self.positives = text_list
        self.pos_embeds = self.encode_text(text_list, self.model.device)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def set_semantics(self, text_list):
        self.semantic_labels = text_list
        self.semantic_embeds = self.encode_text(text_list, self.model.device)
        self.semantic_embeds /= self.semantic_embeds.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        # embed: 3xhxwx512
        n_levels, h, w, c = sem_map.shape
        pos_num = self.semantic_embeds.shape[0]
        phrases_embeds = torch.cat([self.semantic_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(sem_map.dtype)
        sem_pred = torch.zeros(n_levels, h, w)
        for i in range(n_levels):
            output = torch.mm(sem_map[i].view(-1, c), p.T)
            softmax = torch.softmax(10 * output, dim=-1)
            sem_pred[i] = torch.argmax(softmax, dim=-1).view(h, w)
            sem_pred[i][sem_pred[i] >= pos_num] = -1
        return sem_pred.long()

    def get_max_across(self, sem_map): # sem_map: torch.Size([3, 731, 988, 1152]) -> (granuity, h, w, embed_dim)
        '''
        processes a semantic map and returns a relevance map, 
        highlighting the regions of the input image that are most relevant to specific phrases.
        '''
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        
        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1) # 3x731x988x1152 -> 731x988x3x1152 -> 721240x3x1152

        # clip_output = self.pca.transform(clip_output.cpu().numpy()) # ((721240*3), 1152) -> ((721240*3), 512)
        # clip_output = clip_output.view(721240, 3, 512).to(sem_map.device) # (721240, 3, 512)

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j) # clip_output[..., i, :] -> 721240, j -> phrase id
                pos_prob = probs[..., 0:1] # pos_prob -> torch.Size([721240, 1])
                n_phrases_sims[j] = pos_prob # phrase's level relevance score
            n_levels_sims[i] = torch.stack(n_phrases_sims) # each granularity level's relevance score for all phrases
        
        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        return relev_map
