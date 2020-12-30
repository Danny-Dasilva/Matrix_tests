
## Getting started

Begin by cloning the repo and pathing into it

```console
git clone https://github.com/Danny-Dasilva/mkdocs-material.git && cd mkdocs-material 
```



## Install dependencies 

=== "pipenv"

    In case you __do not__ have pipenv you can install it with
    
    ```
    pip install pipenv

    ```

    Install dependencies with 


	```console
	pipenv install

	```
   
    ??? Danger "Warning: Your Pipfile requires python_version 3.8"
        This pipenv file was created using python `3.8` however it is not a requirement to run the package. You can tack 
        on `--python <$your version>` to pipenv in order to run it with that python version. 

        __Example__:

        ```bash
        pipenv install --python 3.7

        ```


=== "docker"

    The [Docker image][1] is a great way to get up and running in a few minutes, as it comes with all dependencies pre-installed. Pull the image for the latest version with:


	```console
	docker pull dannydasilva/mkdocs-material

	```

    The mkdocs executable is provided as an entry point and serve is the default command. 
    If you're not familiar with Docker don't worry, the following sections cover how to serve build and deploy.
    
    The following plugins are bundled with the Docker image:
    
    * markdown-include
    * mkdocs-markdownextradata-plugin

    as well as others
    
 
    ??? Info "How can I add plugins to the Docker image?"
        The docker container bundles useful and common plugins on top of the standard mkdocs
        container however if the plugin you want to use is not included, you can 
        create a new Dockerfile and extend the Docker image with your custom installation routine:

        ```dockerfile
        FROM dannydasilva/mkdocs-material
        RUN pip install ...
        ```

        Next, you can build the image with the following command.

        ```
        docker build -t dannydasilva/mkdocs-material .
        ```

        The new image can be used exactly like the default image, the commands below will operate the same.
   


=== "virtualenv or pip"
    Using a virtual environment is not a requirement but is highly suggested. You can install 
    the dependencies on the system with.

    ``` 
    pip install -r requirements.txt
    ```

    If you would prefer to use `virtualenv`  you can install it with 

    ```
    pip install virtualenv
    ```

    create the environment

    ```
    virtualenv venv
    ```

    Activate the environment

    === "Mac OS / Linux"
        ```
        source venv/bin/activate
        ```
    === "Windows"
        ```
        venv\Scripts\activate
        ```

    and finally install the dependencies
    
    ```
    pip install -r requirements.txt
    ```

    




## Make changes

Feel free to make changes to the index.md or add your own files.
The navigation tree is generated from the `mkdocs.yaml` file. 
Lets say I created `example.md` in `/docs` i could add it to the
site navigation by adding the filename to the .yaml file. 

__in mkdocs.yml__
```yaml linenums="28" hl_lines="10" 
# add navigation here
nav:
- MkDocs: index.md
- features.md
- customization.md
- Example - Nested folder:
  - Index.md with title: example-folder/index.md
  - Sub folder:
      - Custom Title: example-folder/sub-folder/index.md
- example.md
```

###Nesting folders

For larger sections where you need to break markdown files with a similar theme 
you may consider nesting your file structure. Lets say we wanted to add `nested_example.md`
to `docs/example-folder/sub-folder/`. We can add it to the navigation 

```yaml linenums="28" hl_lines="9" 
# add navigation here
nav:
- MkDocs: index.md
- features.md
- customization.md
- Example - Nested folder:
  - Index.md with title: example-folder/index.md
  - Sub folder:
      - Custom Title: example-folder/sub-folder/index.md
      - Nesting folder: example-folder/sub-folder/nested_example.md
- example.md
```

Note how the titles and navigation use the file name by default. You can overide this
with the following syntax `-Custom title: path/to/file.md`




## Run Build and Deploy 

You can run mkdocs locally to preview changes, build out the html files or deploy to github pages with the following commands

###Run

While you are editing your content you can use the live preview feature to see your changes.
The server will automatically rebuild upon saving. By default the server will be hosted at
[localhost:8000](localhost:8000). 

=== "pipenv"
    Quick
    ```
    pipenv run mkdocs serve
    ```


    ??? Info "Work inside env"
        __alternatively__ you can shell into your pipenv instance and serve it there

        ``` 
        pipenv shell
        ```

        then 

        ```
        mkdocs serve
        ```


=== "docker"


    Mount the folder where your mkdocs.yml resides as a volume into /docs:

    * Start development server on [http://localhost:8000](localhost:8000)

    ```
    docker run --rm -it -p 8000:8000 -v ${PWD}:/docs dannydasilva/mkdocs-material
    ```



    * Deploy documentation to GitHub Pages

    ```
    docker run --rm -it -v ~/.ssh:/root/.ssh -v ${PWD}:/docs dannydasilva/mkdocs-material gh-deploy 
    ```




=== "virtualenv or pip" 
    Assuming you are in a virtual environment

    ```
    mkdocs serve
    ```

### Build

When you're finished editing, you can build a static site from your Markdown files with:


=== "pipenv"
    Build Documentation
    ``` 
    pipenv run mkdocs build 
    ```
    

=== "docker"
    Build Documentation
    ``` 
    docker run --rm -it -v ${PWD}:/docs dannydasilva/mkdocs-material build 
    ```

=== "virtualenv or pip"
    Assuming you are in a virtual environment
    ```
    mkdocs build 
    ```


### Deploy 

If you're already hosting your code on GitHub, [GitHub Pages][1] is certainly the most convenient 
way to publish your project documentation. It's free of charge and pretty easy to set up.

  [1]: https://pages.github.com/

=== "pipenv"
    deploy to gihub pages with:
    ```
    pipenv run mkdocs gh-deploy
    ```
    Your documentation should shortly appear at 
    `<$username>.github.io/<$your_github_repository>`.

=== "docker"

    Deploy documentation to GitHub Pages

    ```
    docker run --rm -it -v ~/.ssh:/root/.ssh -v ${PWD}:/docs dannydasilva/mkdocs-material gh-deploy 
    ```

    Your documentation should shortly appear at `<$username>.github.io/<$your_github_repository>`.

=== "virtualenv or pip"
    Assuming you are in a virtual environment
    ```
    mkdocs gh-deploy
    ```
    Your documentation should shortly appear at `<$username>.github.io/<$your_github_repository>`.

