This post contains a list of guidelines to help you write your final report. Your final report does not have to be structured exactly according to the guidelines below, but the content corresponding to each section should appear in some form in your report. This post is meant to supplement the project description on the course website here. Note that the expectations for the final project are significantly higher than prior project deliverables, and grading will be more fine-grained. Sections with longer recommended lengths tend to have more weight in the overall final report grade.

Submission

Please submit your final report as a group submission on Gradescope. The submission deadline is 11:59 p.m. PT, June 5th. No late days can be applied for this. Other than OAE, no exception will be made.

In addition to the report PDF, you must submit the code for your project, especially what was used to obtain the results you are presenting in your final paper and poster. The code will not be graded but it will be considered as evidence of your project's authenticity. No need to worry about formatting and style and whether it is runnable (due to dependencies). Please only include the core code and do not include those code/files that do not directly contribute to the results.

You must also include a Generative AI usage statement in your final report (see Section 8 below). This statement should briefly describe whether and how generative AI tools were used in your project or report writing. For example, you may mention use for brainstorming, debugging, proofreading, code generation, figure/table formatting, or writing assistance. If you did not use generative AI tools, state that explicitly. Use of GenAI is permitted only if consistent with the course and university honor code policies; students remain responsible for the correctness, originality, and attribution of all submitted work.

If you are sharing this project with another class, we require you to submit the report PDF from the other class to this class as well. Remember, it is an honor code violation to use the same final report PDF for multiple classes.

On the submission, we will also ask you to give us consent to publish your projects on the website (this is a great way to showcase and publicize your work).

As a summary/checklist, your Gradescope submission should include:

The final report PDF using the provided template.

The core code used to obtain the results presented in the report and poster.

The report PDF from the other class, if this project is shared with another class.

A completed Contributions & Acknowledgements section.

A Generative AI usage statement.

Formatting

You must use the provided template here. Please use the “final” version as it would properly display the names of the authors. Your final submission must be in PDF format. If you do not use the provided template, we may deduct 1-2 points.

We highly recommend using the LaTeX version with Overleaf. It’s a free, online collaborative platform for editing LaTeX (kind of like Google docs). The given template contains many examples of how to typeset things in LaTeX (e.g. including tables/figures/etc), but if you have any questions please feel free to make an Ed post or visit office hours for help.

You are welcome to adjust the specific sections according to your needs (e.g. combine introduction and related work or separate the experiments from the discussion). However, as stated earlier, the important thing is that each of the key items in each section are still present.

Section 0: Abstract (1-2 paragraphs)

The abstract is required. It should contain a ~1-2 sentence summary of each major section of the paper.

Section 1: Introduction (0.5-1 page)

Explain the problem and why it is important. Discuss your motivation for pursuing this problem. Give some background if necessary. Clearly state what the input and output is. Be very explicit: “The input to our algorithm is a {image, video, patient age, 3D video, etc.}. We then use a {SVM, CNN, GAN, etc.} to output a predicted {age, cancer type, restaurant, ramen, etc.}.” This is very important since different teams have different inputs/outputs spanning different application domains. Being explicit about this makes it easier for readers.

Section 2: Related Work (0.5-1 page)

You should find existing papers, group them into categories based on their approaches, and talk about exemplary ones in each category: Discuss strengths and weaknesses. In your opinion, which approaches were clever/good? What is the state-of-the-art? Do most people perform the task by hand? You should aim to have at least 10 references in the related work. Include previous attempts by others at your problem, previous technical methods, or previous learning algorithms. Google Scholar is very useful for this: https://scholar.google.com/ (search your paper, and you can click “cite” to generate the BibTeX for it.) You can also try http://www.arxiv-sanity.com/ to search for recent arXiv papers.

Section 3: Methods (2 pages)

Describe your learning algorithms, proposed algorithm(s), or theoretical proof(s). Make sure to include relevant mathematical notation, e.g. when you formulate your input(s), output(s) and the loss function(s). It is okay to use formulas from the lecture notes. For each algorithm, give a brief description (2-3 sentences) of how it works. Again, we are looking for your understanding of how these deep learning algorithms work. Although the teaching staff probably know the algorithms, future readers may not (reports will be posted on the class website). Additionally, if you are using a niche or cutting-edge algorithm, you may want to explain your algorithm using several paragraphs. Note: Theory/Algorithms projects may have an appendix showing extended proofs (see appendix description below). Assume the reader has completed CS231N. You don’t need to explain filters and max-pooling, but if you use something like stochastic strides, you should explain that.

If you used an existing codebase and built on top of it, you must state this and explain what you wrote versus what the starter code already came with.

Section 4: Dataset and Features (0.5-1 pages)

Give details about your dataset: How many training/validation/test examples do you have? Is there any data preprocessing you did? What about normalization or data augmentation? What is the resolution of your images? How is your time-series data discretized? Include a citation to where you got your dataset from. Depending on available space, show some examples from your dataset. You should also talk about the features you used. If you extracted features using Fourier transforms, word2vec, histogram of oriented gradients (HOG), PCA, ICA, etc. make sure to talk about it. Try to include examples of your data in the report (e.g. include an image, show a waveform, price graph, etc.).

Section 5: Experiments/Results/Discussion (2-3 pages)

You should also briefly give details about what hyperparameters you chose (e.g. why did you use X learning rate for gradient descent, which optimizer did you pick, what was your mini-batch size and why) and how you chose them. Did you do cross-validation, and if so, how many folds? This should not take more than 1-2 paragraphs. If you want to list more details, please do so in the supplemental material.

Before you list your results, make sure to list and explain what your primary metrics are: accuracy, mAP, inception/mode scores, etc. Provide equations for the metrics if necessary. You can typically find suitable metrics in prior works that study a similar or same topic.

For results, you want to have a mixture of tables and plots. Both quantitative and qualitative results are necessary. To reiterate, you must have both quantitative and qualitative results (unless given prior consent of your TA mentor for certain special cases)! Include visualizations of results, heatmaps, saliency maps, examples of failure cases and a discussion of why certain algorithms failed or succeeded. In addition, explain whether you think you have overfitted to your training set and what, if anything, you did to mitigate that. Make sure to discuss the figures/tables in your main text throughout this section. Your plots should include legends, axis labels, and have font sizes that are readable when printed.

Here’s a list of qualitative & quantitative methods for analysis that might be helpful in your project. None of these are necessary nor will be explicitly looked for by graders – rather, we wanted to provide some (non-exhaustive) guidance on analysis methods:

Saliency maps

Class visualization

t-SNE

Confusion matrices

Common qualitative errors

GANs: compare the generated output to NN in training set (quantitative and qualitative)

GANs: image quality metrics like Inception and Mode scores

VAE: Reporting measures like Annealed Importance Sampling (AIS)

Section 6: Conclusion/Future Work (1-3 paragraphs)

Summarize your report and reiterate key points. Which algorithms were the highest-performing? Why do you think that some algorithms worked better than others? For future work, if you had more time, more team members, or more compute, what would you explore?

Section 7: Appendices

All the content in sections before this point must be between 6 to 8 pages. The Contributions & Acknowledgements, GenAI usage statement, and References/Bibliography do not count toward the 6–8 page limit. Anything critical to your project should appear in the main body, since appendices may not be considered for grading.

This section is optional, as TAs may not consider this section for grading purposes, so anything critical to your project should go into the main paper.

Include additional derivations of proofs which weren’t core to the understanding of your proposed algorithm. Usually, you put equations or other details here when you don’t want to disrupt the flow of the main paper.

Section 8: Contributions & Acknowledgements (not part of page limit)

In this section, you must explicitly state what each person on your team did for the project. This should include contributions to problem formulation, literature review, dataset preparation, implementation, experiments, analysis, writing, figures, and poster preparation, as applicable. If all members contributed equally, you may say so, but you should still briefly describe the main responsibilities of each member.

You must also include a Generative AI usage statement in this section or in a clearly labeled subsection nearby. State whether generative AI tools were used, which tools were used if applicable, and what they were used for. Examples include brainstorming, debugging, proofreading, code generation, figure/table formatting, or writing assistance. If no generative AI tools were used, state this explicitly.

If you are sharing this project with another class, you must include a statement on which part is specifically done for CS231N, and you must also submit the other class PDF along with the CS231N report. If you made use of public code, for example from GitHub, please provide a link to the original repo. Additionally, you must mention any non-CS231N collaborators and include a brief sentence on what they did for your project. See the AlphaGo paper’s contributions statement for an example. If you are part of a research lab and made use of their job scheduling, containerization, or GPUs, briefly include a sentence description on this as well.

Section 9: References/Bibliography (No page limit)

This section should include citations for: (1) Any papers mentioned in the related work section. (2) Papers describing algorithms that you used which were not covered in class. (3) Code or libraries you downloaded and used. This includes libraries such as scikit-learn, TensorFlow, PyTorch, etc. For simplicity, please use the provided BibTeX file in the template (see our "Section 2" suggestions for how to easily get BibTeX citations from Google Scholar). Main body text, figures, and any discussions are strictly forbidden from this section. We are excluding the references section from the page limit to encourage students to perform a thorough literature review/related work section without being space-penalized if they include more references. The citations must be in the same format as provided in the template. We will deduct points if they are missing or have inconsistent formatting.