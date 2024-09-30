Hallo zusammen,

ich habe mal wieder ein Python Problem an dem ich verzweifle. Ich bin mir nicht mal sicher, ob das überhaupt der pythonic way ist das zu tun. Also bin ich für Vorschläge und Ratschläge jeglicher Art offen.

Ich versuche das Problem mal herauszuarbeiten:

Es gibt eine Klasse `Dataset`:
```python
class Dataset[Data]:
  @abstractmethod
  def get_data(self) -> Data:
    pass
```
Diese hat Daten vom Typ `Data`. Ist also generisch über die Daten die sie liefern kann. Jetzt gibt es ganz viele verschiedene Child-Classes die von dieser `Dataset` Klasse erben und entsprechend `get_data` overriden. Die child-classes fügen natürlich ggf. auch eigene Methoden hinzu...

Bspw:
```python
class MyPolarsData(Dataset[polars.DataFrame]):
  data: polars.DataFrame

  @override
  def get_data(self) -> polars.DataFrame:
    return data

  # some arbitrary additional methods
  def foo(self) -> int:
    return len(data)
```

und es gibt halt Methoden, die diese `Dataset` objekte entgegennehmen:

```python
def do_something[Inner](dataset: Dataset[Inner]) -> None:
  dataset.get_data()
  # ...
```

Soweit so gut. Ich möchte nun eine Funktionalität hinzufügen, die die Daten aus einer `Dataset` mapped. Also ein callable applied auf den Daten bevor es von `get_data()` returned wird. Mir schwebt sowas vor:

```python
def my_fancy_function(data: polars.DataFrame) -> polars.DataFrame:
  # modify data
  return data

baz = MyPolarsData(...)
foobar = baz.with_map(my_fancy_function)
mapped_data = foobar.get_data() # always calls my_fancy_function on the data before returning it
```

Wie drückt man sowas aus? Ich dachte erst, ich könnte eine Art wrapper class bauen, mit einem `inner` object und dann halt die `get_data` entsprechend implementieren. Aber dann verliere ich alle methoden von der eigentlichen Klasse. Ich möchte weiterhin `foo()` callen können...
Und was mir auch immer wieder alles verkompliziert, ich möchte das am liebsten "type-safe" (im Python Kontext).

Vielleicht ist da auch grundlegend konzeptionell was falsch. Ich bin über jegliche Gedanken und Kommentare froh. Hat jemand Ideen?
