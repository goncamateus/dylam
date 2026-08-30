# TMLR Beyond PDF Author Kit


Please see the submission instructions at https://tmlr-beyond-pdf.org/submission-process for details.


## Testing Your Submission Locally

To render your website locally, you must first have [Docker](https://www.docker.com/) installed. Then, you can run:

```
python compile_submission.py
```

This will build your website and start a server so that you can view the resulting page locally at [http://0.0.0.0:8080/tmlr-beyond-pdf/under_review/submission/](http://0.0.0.0:8080/tmlr-beyond-pdf/under_review/submission/).


## Submitting to TMLR on OpenReview

Once you are happy with your submission and have tested it locally, please zip the `submission_folder` using the command:

```
zip -r submission_folder.zip submission_folder
```

Then, create a submission as usual on [OpenReview](https://openreview.net/group?id=TMLR), uploading `submission_folder.zip` in the "Beyond PDF" section. Please select "Beyond PDF Submission" as the Submission Type. Once you submit to OpenReview, the system will automatically pull the zipped folder from your submission and add it to the [Under Review page of the TMLR Beyond PDF website](https://tmlr-beyond-pdf.org/under_review/). The reviewing process will take place on OpenReview as usual, with reviewers referencing your compiled page on the TMLR Beyond PDF website.

### PDF for OpenReview

The submission on OpenReview also needs to have a PDF for archival purposes.

**Please use the "Print to PDF" feature in the browser to download your PDF version. Do not use any other method to make your PDF.**
