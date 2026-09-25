# Experiment comparison

## Training configuration

| experiment | params | precision | seq | eff. batch | lr | steps | tokens | train time | status |
|---|---|---|---|---|---|---|---|---|---|
| exp_001_tiny_tinystories | 4.95M | fp32 | preset | 64 | 0.001 | 3000 | 49.2M | 78.8 min | completed |
| exp_002_sft_tinystories | 4.95M | fp32 | preset | 32 | 0.0003 | 600 | 4.6M | 6.5 min | completed |
| exp_003_sft_tinystories_long | 4.95M | fp32 | preset | 32 | 0.0003 | 2000 | 16.2M | 24.8 min | completed |

## Validation and evaluation metrics

| experiment | checkpoint | val loss | val ppl | test ppl | top-1 | probes | distinct-2 (sampled) | loops | non-words | edge cases |
|---|---|---|---|---|---|---|---|---|---|---|
| exp_001_tiny_tinystories | best | 1.644 | 5.18 | 5.09 | 0.594 | 0.927 | 0.538 | 0.000 | 0.001 | 14/14 |
| exp_002_sft_tinystories | best | 1.421 | 4.14 | 4.02 | 0.636 | 0.873 | 0.538 | 0.000 | 0.002 | 14/14 |
| exp_003_sft_tinystories_long | best | 1.441 | 4.22 | 4.09 | 0.635 | 0.873 | 0.546 | 0.000 | 0.001 | 14/14 |

## Generation samples (greedy, first prompts)

### exp_001_tiny_tinystories

> **Once upon a time, there was a little girl named Lily.** She had a big, red ball that she loved to play with. One day, she saw a small, sad bird sitting on a branch. The bird was sad because it could not fly. Lily wanted to help the bird. She thought of a plan. She found a long stick and used it to push the bird out of the tree. The bird was very happy and said, "Thank you, Lily!" Lily and the bird played together all day. They ran, jumped, and laughed.…

> **Once upon a time, there was a little girl named Tom.** Tom was very excited because he had a new toy. He wanted to play with it. He asked his mom, "Can I play with the toy, please?" His mom said, "Yes, but be careful. Don't go too close to the toy." Tom was very happy. He played with the toy all day. But then, something unexpected happened. The toy started to move! Tom was scared. He didn't know what to do. He looked around and saw a big, friendly dog…

> **Once upon a time, there was a little girl named Mia.** Mia loved to play with her toys. One day, she found a big box in her room. She was very happy and wanted to see what was inside. Mia opened the box and found a big, soft pillow. She was very happy. She put the pillow on her bed and went to sleep. But when she woke up, she saw that the pillow was not a pillow. It was a big, soft pillow! Mia was very surprised. She did not know what to do. She thoug…

### exp_002_sft_tinystories

> **Tell me a story about Lily.** Once upon a time, there was a little girl named Lily. She loved to play with her toys and have fun. One day, she found a big, shiny rock in her yard. She picked it up and showed it to her mom. "Look, Mom! I found a rock!" Lily said. Her mom smiled and said, "That's a nice rock, Lily. Let's play with it." Lily and her mom played with the rock all day. They made it go up and down, and then they went…

> **Tell me a story about Tom.** Once upon a time, there was a little boy named Tom. Tom loved to play with his toy cars. One day, he found a big, shiny rock. He was very happy and wanted to show his friend, Sam. Tom went to Sam's house and said, "Look at my new toy car! It is so big and shiny!" Sam looked at the shiny rock and said, "Wow, that's a cool rock! Let's play with it!" They played with the rock and had lots of fun. But…

> **Tell me a story about Mia.** Once upon a time, there was a little girl named Mia. Mia loved to play with her toys and eat yummy food. One day, Mia found a big, red apple on the ground. She picked it up and took it home. Mia's mom saw her and said, "Mia, you need to eat your apple. It is not good to eat." Mia felt sad and said, "I am sorry, Mom. I will not eat it." Mia's mom gave her a big, red apple to eat. Mia was happy and …

### exp_003_sft_tinystories_long

> **Tell me a story about Lily.** Once upon a time, there was a little girl named Lily. She had a big, soft bed in her room. Lily loved to play in her room with her toys. One day, she found a small, soft pillow. She was very happy and started to play with it. Lily's mom saw her playing with the pillow and said, "Lily, you need to clean your room. You need to clean your room and put your toys in the pillow." Lily listened to her mo…

> **Tell me a story about Tom.** Once upon a time, there was a little boy named Tom. Tom had a toy car that he loved to play with. One day, Tom's toy car broke. Tom was sad and didn't know what to do. Tom's mom saw him and asked, "Why are you sad, Tom?" Tom said, "My car is broken, and I can't play with it." His mom hugged him and said, "Don't worry, Tom. We will fix it together." Tom's mom fixed the broken car with some glue and…

> **Tell me a story about Mia.** Once upon a time, there was a little girl named Mia. Mia loved to play with her toys. One day, she found a big box of crayons. She was very happy and started to draw. Mia's mom saw her and said, "Mia, you need to clean your room. You need to clean your room." Mia felt sad and said, "I will, Mom, but I will help you clean." Mia's mom took her to the kitchen. Mia found some soft cloth and wiped her …

