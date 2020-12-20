
## Basics

Ideally you have a basic understanding of markdown syntax however if you 
need a refresher feel free to visit [here][1]. The following are extra features
built into mkdocs

##Bash terminal

Using `termynal.js` and `termynal.css` in `/docs/js` and `/docs/js` respectively we can show
terminal output. By wraping a `console` code block in a `div` tag with `class="termy"` you can 
display a loading promt as well as example output in your markdown files. This termynal styling
is provided courtesy of <a href="https://github.com/tiangolo/fastapi/tree/master/docs" class="external-link" target="_blank">FastApi</a>


_Example_:

````markdown
<div class="termy">

```console
$ some_example_command.sh

---> 100%
INFO:     Started server process [28722]
INFO:     Application startup complete.
```

</div>
````


_Result_:


<div class="termy">

```console
$ some_example_command.sh

---> 100%
INFO:     Started server process [28722]
INFO:     Application startup complete.
```

</div>


##Code Blocks
[:octicons-file-code-24: Source][2] · [:octicons-workflow-24: Extension][3] ·
:octicons-zap-24: Supersedes: [CodeHilite][4]

The [Highlight][3] extension, which is part of [Python Markdown Extensions][5],
integrates with Material for MkDocs and provides several options for
configuring syntax highlighting of code blocks, to see how to set up and configure this
see the <a href="https://squidfunk.github.io/mkdocs-material/reference/code-blocks/" class="external-link" target="_blank">mkdocs-material documentation</a> 

    
### Usage

Code blocks must be enclosed with two separate lines containing three backticks.
To add code highlighting to those blocks, add the language short name directly
after the opening block. See the [list of available lexers][17] to find the
short name for a given language.

_Example_:

```` markdown
``` python
import tensorflow as tf
```
````

_Result_:

``` python
import tensorflow as tf
```

### Adding line numbers

Line numbers can be added to a code block by using the `linenums="<start>"`
option directly after the short name, whereas `<start>` represents the starting
line number. A code block can start from a line number other than `1`, which
allows splitting large code blocks for readability.

_Example_:

```` markdown hl_lines="1"
``` python linenums="1"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```
````

_Result_:

``` python linenums="1"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```

??? Info "Starting at a different line number"
    The number passed into `#!python range()` will be the start of the line
    _Example_:

    ```` markdown hl_lines="1"
    ``` python linenums="7"
    def bubble_sort(items):
        for i in range(len(items)):
            for j in range(len(items) - 1 - i):
                if items[j] > items[j + 1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
    ```
    ````

    _Result_:

    ``` python linenums="7"
    def bubble_sort(items):
        for i in range(len(items)):
            for j in range(len(items) - 1 - i):
                if items[j] > items[j + 1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
    ```


### Highlighting specific lines

Specific lines can be highlighted by passing the line numbers to the `hl_lines`
argument placed right after the language short name. Note that line counts start
at `1`, regardless of the starting line number specified as part of `linenums`.

_Example_:

```` markdown
``` python hl_lines="2 3"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```
````

_Result_:

``` python hl_lines="2 3"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```

??? Tip "Highlighting inline code"
    With [InlineHilite][18] enabled, inline code blocks can be highlighted by
    prefixing them with a shebang-like sequence, i.e. `#!`, directly followed by
    the short language name [17].

    _Example_:

    ``` markdown
    The `#!python range()` function is used to generate a sequence of numbers.
    ```

    _Result_:

    The `#!python range()` function is used to generate a sequence of numbers.

    [18]: #inlinehilite

##Admonitions
[:octicons-file-code-24: Source][5] · [:octicons-workflow-24: Extension][6]

Admonitions, also known as call-outs, are an devent choice for including side content without significantly interrupting the document flow. 
Material for MkDocs provides several different types of admonitions and allows for the inclusion and nesting of arbitrary content. A
details block follows the syntax and semantics of admonition blocks, but must start with `???`. The `???` can then be followed by a
admonition type e.g. `example` 
<a href="https://squidfunk.github.io/mkdocs-material/reference/admonitions/#supported-types" class="external-link" target="_blank">(view supported types)</a>. 

### Collapsed Blocks

_Example_:

``` markdown
??? example 
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.
```

_Result_:

??? example 

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

### Open blocks

You can add a `"title"` by adding quotations after your supported admonition type
<a href="https://squidfunk.github.io/mkdocs-material/reference/admonitions/#supported-types" class="external-link" target="_blank">(view supported types)</a>. 
Adding a `+` after `???` will render the block as open on page load:

_Example_:

``` markdown
???+ note "Example Title"
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.
```

_Result_:

???+ note "Example Title"

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

There are many supported admonition types and these can be added to explain bugs or issues users may encounter. 
You can fit code blocks or any other mkdocs markdown inside these blocks so feel free to use them to add collapsible content.

??? Tip "No title Icon or Bar"

    Similar to changing the title, the icon and title can be omitted entirely by adding an empty string directly after the type qualifier. 
    Note that this will not work for collapsible blocks.

	``` markdown
	!!! note ""
	    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
	    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
	    massa, nec semper lorem quam in massa.
	```

	_Result_:

	!!! note ""

	    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
	    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
	    massa, nec semper lorem quam in massa.



## Images

You can add images using the following syntax `![Placeholder_text](path/to/img.jpg)`.
You can add images in the `docs/img` folder and reference them in markdown. When the [Attribute List][7] extension is enabled, 
images can be aligned by adding the respective alignment directions via the `align` attribute, i.e.
`align=left` or `align=right`. Along with this you can assign width with the syntax `width=200`.

### Alignment 

=== "Left"

    _Example_:

    ``` markdown
    ![Placeholder](img/example.png){: align=left width=300 }
    ```

    _Result_:

    ![Placeholder](img/example.png){: align=left width=300 }

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

=== "Right"

    _Example_:

    ``` markdown
    ![Placeholder](https://dummyimage.com/600x400/eee/aaa){: align=right }
    ```

    _Result_:

    ![Placeholder](https://dummyimage.com/600x400/f5f5f5/aaaaaa&text=–%20Image%20–){: align=right width=300 }

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

_If there's insufficient space to render the text next to the image, the image
will stretch to the full width of the viewport, e.g. on mobile viewports._

### Image captions

Sadly, the Markdown syntax doesn't provide native support for image captions,
but it's always possible to resort to HTML. Using `figure` and `figcaption`, captions can be added to images.

_Example_:

```html
<figure>
  <img src="https://dummyimage.com/600x400/eee/aaa" width="300" />
  <figcaption>Image caption</figcaption>
</figure>
```

_Result_:

<figure>
  <img src="https://dummyimage.com/600x400/f5f5f5/aaaaaa&text=–%20Image%20–" width="300" />
  <figcaption>Image caption</figcaption>
</figure>



??? Tip "Image lazy-loading"

	Modern browsers provide [native support for lazy-loading images][8] through the
	`loading` attribute, which degrades to eager-loading in browsers without
	support. As with [image alignment][9], if the [Attribute List][7] extension is
	enabled, images can be lazy-loaded by adding `loading=lazy`.

	_Example_:

	``` markdown
	![Placeholder](https://dummyimage.com/600x400/eee/aaa){: loading=lazy }
	```

## Content tabs

Sometimes, it's desirable to group alternative content under different tabs, e.g. when describing how to access an API from different languages or environments. Material for MkDocs allows for beautiful and functional tabs, grouping code blocks and other content
<a href="https://squidfunk.github.io/mkdocs-material/reference/content-tabs/" class="external-link" target="_blank">see example docs</a>.

_Example_:

```
=== "C"

    ``` c
    #include <stdio.h>

    int main(void) {
      printf("Hello world!\n");
      return 0;
    }
    ```

=== "C++"

    ``` c++
    #include <iostream>

    int main(void) {
      std::cout << "Hello world!" << std::endl;
      return 0;
    }
    ```
```

_Result_:

=== "C"

    ``` c
    #include <stdio.h>

    int main(void) {
      printf("Hello world!\n");
      return 0;
    }
    ```

=== "C++"

    ``` c++
    #include <iostream>

    int main(void) {
      std::cout << "Hello world!" << std::endl;
      return 0;
    }
    ```


??? Tip "Grouping other content"
    With [InlineHilite][18] enabled, inline code blocks can be highlighted by
    prefixing them with a shebang-like sequence, i.e. `#!`, directly followed by
    the short language name [17].

    _Example_:

    ````markdown
    === "Example Content"

        ### Lorem ipsum dolor sit amet
        Lorem ipsum dolor sit amet

        `Lorem ipsum dolor sit amet`

    === "Ordered list"

        1. Sed sagittis eleifend rutrum
        2. Donec vitae suscipit est
        3. Nulla tempor lobortis orci
    ````

    _Result_:

    === "Example Content"

        ### Lorem ipsum dolor sit amet
        Lorem ipsum dolor sit amet

        `Lorem ipsum dolor sit amet`

    === "Ordered list"

        1. Sed sagittis eleifend rutrum
        2. Donec vitae suscipit est
        3. Nulla tempor lobortis orci




## Icons

Material for mkdocs supports [more than 7,000 icons][7] and 
thousands of emojis with very little effort. Material design,
font awesome and octicons icons are supported

Emojis and Icons can be integrated in Markdown by putting the shortcode of the emoji
between two colons. To see how to add custom css rules to Icons as well as custom Icons see the 
<a href="https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/" class="external-link" target="_blank">mkdocs icon documentation</a>.

_Example_:

````markdown
:smile: 

:material-account-circle: – `.icons/material/account-circle.svg`

:fontawesome-regular-laugh-wink: – `.icons/fontawesome/regular/laugh-wink.svg`

:octicons-octoface-16: – `.icons/octicons/octoface-16.svg`
````

_Result_:

:smile: 

:material-account-circle: – `.icons/material/account-circle.svg`

:fontawesome-regular-laugh-wink: – `.icons/fontawesome/regular/laugh-wink.svg`

:octicons-octoface-16: – `.icons/octicons/octoface-16.svg`

## Data tables

Data tables can be used at any position in your project documentation and can
contain arbitrary Markdown, including inline code blocks, as well as [icons and
emojis][11].

_Example_:

``` markdown
| Method      | Description                          |
| ----------- | ------------------------------------ |
| `GET`       | :material-check:     Fetch resource  |
| `PUT`       | :material-check-all: Update resource |
| `DELETE`    | :material-close:     Delete resource |
```

_Result_:

| Method      | Description                          |
| ----------- | ------------------------------------ |
| `GET`       | :material-check:     Fetch resource  |
| `PUT`       | :material-check-all: Update resource |
| `DELETE`    | :material-close:     Delete resource |

  [3]: icons-emojis.md

## Tasklists

When the [Tasklist][12] extension is enabled, unordered list items can be
prefixed with `[ ]` to render an unchecked or `[x]` to render a checked
checkbox. Along with normal markdown lists which can be prefficed with `*`
or `1.` tasklists provide you a way of annotating completed tasks.

_Example_:

``` markdown
* [x] Lorem ipsum dolor sit amet, consectetur adipiscing elit
* [ ] Vestibulum convallis sit amet nisi a tincidunt
    * [x] In hac habitasse platea dictumst
    * [x] In scelerisque nibh non dolor mollis congue sed et metus
    * [ ] Praesent sed risus massa
* [ ] Aenean pretium efficitur erat, donec pharetra, ligula non scelerisque
```

_Result_:

* [x] Lorem ipsum dolor sit amet, consectetur adipiscing elit
* [ ] Vestibulum convallis sit amet nisi a tincidunt
    * [x] In hac habitasse platea dictumst
    * [x] In scelerisque nibh non dolor mollis congue sed et metus
    * [ ] Praesent sed risus massa
* [ ] Aenean pretium efficitur erat, donec pharetra, ligula non scelerisque

  [7]: #tasklist

##Footnotes and Links 

Footnotes are a great way to add references to supplemental or additional information for a specific section of a document without interrupting the document flow.
Material for MkDocs provides the ability to insert inline footnotes and render them at the bottom of the page. 
A footnote reference must be enclosed in square brackets and must start with a
caret `^`, directly followed by an arbitrary identifier, which is similar to
the standard Markdown link syntax. 
Along with this you can just use an arbitrary identifier in square brackets to insert a link.
_Example_:

``` markdown
Lorem ipsum[^1] dolor sit amet, consectetur adipiscing elit.[^2]

Lorem ipsum[1] dolor sit amet, consectetur adipiscing elit[2].
```

_Result_:

Lorem ipsum[^1] dolor sit amet, consectetur adipiscing elit.[^2]

Lorem [ipsum][1] dolor sit amet, consectetur adipiscing [elit][2].


### Adding footnote content

The footnote content must be declared with the same identifier as the reference.
It can be inserted at an arbitrary position in the document and is always
rendered at the bottom of the page. Furthermore, a backlink to the footnote
reference is automatically added.

#### on a single line

Short statements can be written on the same line.

_Example_:

``` markdown
[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[2]: https://github.com/squidfunk/mkdocs-material/blob/master/src/assets/stylesheets/main/extensions/pymdownx/_highlight.scss
```

_Result_:

[Jump to footnote at the bottom of the page](#fn:1)

Or click the link on the highlighted words to follow the link


  [^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
  [1]: https://guides.github.com/features/mastering-markdown/  
  [2]: https://github.com/squidfunk/mkdocs-material/blob/master/src/assets/stylesheets/main/extensions/pymdownx/_highlight.scss
  [3]: https://facelessuser.github.io/pymdown-extensions/extensions/highlight/
  [4]: https://python-markdown.github.io/extensions/code_hilite/
  [5]: https://github.com/squidfunk/mkdocs-material/blob/master/src/assets/stylesheets/main/extensions/markdown/_admonition.scss
  [6]: https://python-markdown.github.io/extensions/admonition/
  [7]: https://python-markdown.github.io/extensions/attr_list/ 
  [8]: https://caniuse.com/loading-lazy-attr
  [9]: #alignment 
  [10]: https://github.com/squidfunk/mkdocs-material/tree/master/material/.icons
  [11]: #icons
  [12]: https://facelessuser.github.io/pymdown-extensions/extensions/tasklist/ 
  [17]: https://pygments.org/docs/lexers/
  [18]: https://squidfunk.github.io/mkdocs-material/reference/code-blocks/#inlinehilite
