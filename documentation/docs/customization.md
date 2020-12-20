

# Changing the colors

As any proper Material Design implementation, Material for MkDocs supports
Google's original [color palette][1], which can be easily configured through 
`mkdocs.yml`. Furthermore, colors can be customized with a few lines of CSS to
fit your brand's identity by using [CSS variables][2].

  [1]: http://www.materialui.co/colors
  [2]: #custom-colors

## Configuration

### Color palette

#### Color scheme

[:octicons-file-code-24: Source][3] · :octicons-milestone-24: Default: `default`

Material for MkDocs supports two _color schemes_: a light mode, which is just
called `default`, and a dark mode, which is called `slate`. The color scheme
can be set via `mkdocs.yml`:

``` yaml hl_lines="3"
theme:
  palette:
    scheme: default
```

:material-cursor-default-click-outline: Click on a tile to change the color
scheme:

<div class="tx-switch">
  <button data-md-color-scheme="default"><code>default</code></button>
  <button data-md-color-scheme="slate"><code>slate</code></button>
  <button data-md-color-scheme="iceberg"><code>iceberg</code></button>
</div>

<script>
  var buttons = document.querySelectorAll("button[data-md-color-scheme]")
  buttons.forEach(function(button) {
    button.addEventListener("click", function() {
      var attr = this.getAttribute("data-md-color-scheme")
      document.body.setAttribute("data-md-color-scheme", attr)
      var name = document.querySelector("#__code_0 code span:nth-child(7)")
      name.textContent = attr
    })
  })
</script>
#### Primary color

[:octicons-file-code-24: Source][4] · :octicons-milestone-24: Default: `indigo`

The _primary color_ is used for the header, the sidebar, text links and several
other components. In order to change the primary color, set the following value
in `mkdocs.yml` to a valid color name:

``` yaml hl_lines="3"
theme:
  palette:
    primary: indigo
```

:material-cursor-default-click-outline: Click on a tile to change the primary
color:

<div class="tx-switch">
  <button data-md-color-primary="red"><code>red</code></button>
  <button data-md-color-primary="pink"><code>pink</code></button>
  <button data-md-color-primary="purple"><code>purple</code></button>
  <button data-md-color-primary="deep-purple"><code>deep purple</code></button>
  <button data-md-color-primary="indigo"><code>indigo</code></button>
  <button data-md-color-primary="blue"><code>blue</code></button>
  <button data-md-color-primary="light-blue"><code>light blue</code></button>
  <button data-md-color-primary="cyan"><code>cyan</code></button>
  <button data-md-color-primary="teal"><code>teal</code></button>
  <button data-md-color-primary="green"><code>green</code></button>
  <button data-md-color-primary="light-green"><code>light green</code></button>
  <button data-md-color-primary="lime"><code>lime</code></button>
  <button data-md-color-primary="yellow"><code>yellow</code></button>
  <button data-md-color-primary="amber"><code>amber</code></button>
  <button data-md-color-primary="orange"><code>orange</code></button>
  <button data-md-color-primary="deep-orange"><code>deep orange</code></button>
  <button data-md-color-primary="brown"><code>brown</code></button>
  <button data-md-color-primary="grey"><code>grey</code></button>
  <button data-md-color-primary="blue-grey"><code>blue grey</code></button>
  <button data-md-color-primary="black"><code>black</code></button>
  <button data-md-color-primary="white"><code>white</code></button>
  <button data-md-color-primary="iceberg_blue"><code>iceberg_blue </code></button>
</div>
<style>
[data-md-color-primary="iceberg_blue"] {
--md-primary-fg-color: red
}
</style>
<script>
  var buttons = document.querySelectorAll("button[data-md-color-primary]")
  buttons.forEach(function(button) {
    button.addEventListener("click", function() {
      var attr = this.getAttribute("data-md-color-primary")
      document.body.setAttribute("data-md-color-primary", attr)
      var name = document.querySelector("#__code_2 code span:nth-child(7)")
      name.textContent = attr.replace("-", " ")
    })
  })
</script>

  [4]: https://github.com/squidfunk/mkdocs-material/blob/master/src/assets/stylesheets/palette/_primary.scss

#### Accent color

[:octicons-file-code-24: Source][5] · :octicons-milestone-24: Default: `indigo`

The _accent color_ is used to denote elements that can be interacted with, e.g.
hovered links, buttons and scrollbars. It can be changed in `mkdocs.yml` by
choosing a valid color name:

``` yaml hl_lines="3"
theme:
  palette:
    accent: indigo
```

:material-cursor-default-click-outline: Click on a tile to change the accent
color:

<style>
  .md-typeset button[data-md-color-accent] > code {
    background-color: var(--md-code-bg-color);
    color: var(--md-accent-fg-color);
  }
</style>

<div class="tx-switch">
  <button data-md-color-accent="red"><code>red</code></button>
  <button data-md-color-accent="pink"><code>pink</code></button>
  <button data-md-color-accent="purple"><code>purple</code></button>
  <button data-md-color-accent="deep-purple"><code>deep purple</code></button>
  <button data-md-color-accent="indigo"><code>indigo</code></button>
  <button data-md-color-accent="blue"><code>blue</code></button>
  <button data-md-color-accent="light-blue"><code>light blue</code></button>
  <button data-md-color-accent="cyan"><code>cyan</code></button>
  <button data-md-color-accent="teal"><code>teal</code></button>
  <button data-md-color-accent="green"><code>green</code></button>
  <button data-md-color-accent="light-green"><code>light green</code></button>
  <button data-md-color-accent="lime"><code>lime</code></button>
  <button data-md-color-accent="yellow"><code>yellow</code></button>
  <button data-md-color-accent="amber"><code>amber</code></button>
  <button data-md-color-accent="orange"><code>orange</code></button>
  <button data-md-color-accent="deep-orange"><code>deep orange</code></button>
</div>

<script>
  var buttons = document.querySelectorAll("button[data-md-color-accent]")
  buttons.forEach(function(button) {
    button.addEventListener("click", function() {
      var attr = this.getAttribute("data-md-color-accent")
      document.body.setAttribute("data-md-color-accent", attr)
      var name = document.querySelector("#__code_3 code span:nth-child(7)")
      name.textContent = attr.replace("-", " ")
    })
  })
</script>

  [5]: https://github.com/squidfunk/mkdocs-material/blob/master/src/assets/stylesheets/palette/_accent.scss
### custom css


## Logo
The logo can be changed in the `mkdocs.yaml` file
__in mkdocs.yml__
```yaml linenums="13" hl_lines="2" 
  #change logo here
  logo: img/logo-white.svg
  favicon: img/favicon.png
 
```



## Additional CSS and Javascript

=== "CSS"

    If you want to tweak some colors or change the spacing of certain elements, 
    you can do this in a separate stylesheet. The easiest way is by creating a 
    new stylesheet file in the `docs/` directory:
    ```
    ├─ docs/
    │  └─ stylesheets/
    │     └─ extra.css
    └─ mkdocs.yml
    ```
    Then, add the following line to mkdocs.yml:

    ```yaml
    extra_css:
      - stylesheets/extra.css
    ```

=== "Javascript"
    The same is true for additional JavaScript. If you want to integrate 
    another syntax highlighter or add some custom logic to your theme, 
    create a new JavaScript file in the `docs/` directory:
    ```
    ├─ docs/
    │  └─ javascripts/
    │     └─ extra.js
    └─ mkdocs.yml
    ```
    Then, add the following line to mkdocs.yml:

    ```yaml
    extra_css:
      - javascripts/extra.js
    ```

## Styling code blocks

=== "Python"

    ```python
	from __future__ import annotations
	from datetime import date

	# random Person
	class Person:
	    def __init__(self, name: str, age: int):
			self.name = name
			self.age = age

	    @classmethod
	    def fromBirthYear(cls, name: str, birthYear: int) -> Person:
			return cls(name, date.today().year - birthYear)

	    def __repr__(self) -> str:
			return f"{self.name}'s age is: {str(self.age)}"

	person = Person('Adam', 19)
	print(person)

	person1 = Person.fromBirthYear('John',  1985)
	print(person1)
    ```

=== "Javacript"

    ```js
	import '/modules/my-module.js';

	class Person {
	  constructor(name, age) {
	    this._name = name;
	    this._age = age;
	  }
	  //unused Method
	  increaseAge() {
	    this._age += 1;
	  }
	}
	var fromBirthYear = (name, age) => {
	  var year = new Date().getFullYear()
	  return new Person(name, year-age)

	}

	const person = new Person("Adam", 19)
	console.log(person)

	const person1 = fromBirthYear('John',  1985)
	console.log(person1)
    ```

=== "HTML"

    ```html
	<!DOCTYPE html>
	<html lang="en">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>Document</title>
		<!-- CSS -->
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css"
			crossorigin="anonymous">
	</head>
	<body>
		<picture>
			<source type="image/avif" srcset="https://picture-demo.netlify.app/photo.avif">
			<source type="image/webp" srcset="https://picture-demo.netlify.app/photo.webp">
			<img src="https://picture-demo.netlify.app/photo.jpg" alt="Red Panda">
		</picture>
	</body>
	<!-- jQuery and JS bundle w/ Popper.js -->
	<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" crossorigin="anonymous" async></script>
	<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/js/bootstrap.bundle.min.js" crossorigin="anonymous"
		async></script>
	</html>
    ```

=== "CSS"

    ```css
	:root {
	  --main-bg-color: red;
	}
	.one {
	  color: white;
	  background-color: var(--main-bg-color);
	  margin: 10px;
	  width: 50px;
	  height: 50px;
	  display: inline-block;
	}

	.two {
	  color: white;
	  background-color: var(--main-bg-color);
	  margin: 10px;
	  width: 150px;
	  height: 70px;
	  display: inline-block;
	}
    ```


=== "Bash"

    ```bash
	#!/bin/bash
	ARG=$1

	function mimic {
	  if [[ -z $ARG ]]; then
	    ARG='world'
	  fi
	  echo "hello $ARG"
	}

	mimic $ARG

    ```

=== "Java"

    ```Java
	public class MyClass {
	  int x = 5;

	  public static void main(String[] args) {
	    MyClass myObj = new MyClass();
	    System.out.println(myObj.x);
	  }
	}
    ```

=== "C"

    ```C
	int main(void)
	{
	  RectangleClass r1;

	  rectangle_new_with_lengths(&r1, 4.f, 5.f);
	  printf("rectangle r1's area is %f units square\n", shape_computeArea(&r1));
	  return 0;
	}
    ```

=== "Go"

    ```go
	package employee

	import (  
	    "fmt"
	)

	type employee struct {  
	    firstName   string
	    lastName    string
	    totalLeaves int
	    leavesTaken int
	}

	func New(firstName string, lastName string, totalLeave int, leavesTaken int) employee {  
	    e := employee {firstName, lastName, totalLeave, leavesTaken}
	    return e
	}

	func (e employee) LeavesRemaining() {  
	    fmt.Printf("%s %s has %d leaves remaining\n", e.firstName, e.lastName, (e.totalLeaves - e.leavesTaken))
	}
    ```

